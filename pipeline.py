"""
pipeline.py — Unified SSV2A Pipeline: Full Gradio Interface

Complete end-to-end pipeline with step-by-step visibility:
  Step 1: Upload image → YOLO detection (toggle objects ON/OFF)
  Step 2: Auto-chain: LLM filter → BLIP → CLIP embeddings + SAM (3D only)
  Step 3: Generate 3D World (Hunyuan3D meshes + scene)
  Step 4: Generate Spatial Audio + camera re-render

Launch:
    python pipeline.py --port 7875
"""

from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as _np
if not hasattr(_np, "bool8"):
    _np.bool8 = _np.bool_
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64
if not hasattr(_np, "obj2sctype"):
    def _obj2sctype(rep, default=None):
        try:
            return _np.dtype(rep).type
        except Exception:
            return default
    _np.obj2sctype = _obj2sctype

# ── HF cache 
_user = os.getenv("USER") or os.getenv("LOGNAME") or "unknown"
_hf_cache = f"/Data/{_user}/hf_cache/hub"
os.makedirs(_hf_cache, exist_ok=True)
os.environ.setdefault("HF_HUB_CACHE", _hf_cache)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _hf_cache)
os.environ.setdefault("HF_HOME", f"/Data/{_user}/hf_cache")

import argparse
import copy
import json
import math
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont

# ── SSV2A core
from ssv2a.data.detect_gemini_SAM import detect as detect_yolo_4tuple
from ssv2a.data.utils import clip_embed_images, emb2seq, save_wave, set_seed
from ssv2a.model.pipeline import Pipeline, image_to_audio as i2a_image_to_audio

# ── Depth estimation 
from ssv2a.data.depth_estimation import (
    load_depth_model, estimate_depth, depth_for_bboxes,
)

# ── 3D spatial 
from ssv2a.model.spatial_3d import (
    CameraIntrinsics, Object3D, backproject_objects,
    apply_listener_rotation, compute_3d_spatial_params, SAMPLE_RATE,
)

# ── 3D world generation 
from ssv2a.model.generate_3d_scene import (
    generate_object_mesh, _boost_vertex_colors, create_grid_platform,
    build_3d_world, export_scene_glb, export_individual_glbs, export_scene_json,
    segment_object, unload_sam, load_blip, unload_blip, caption_crop,
)

# ── Shared viewer helpers (progress bar, viewer scene, HTML) ─────────────
from gradio_3d_world import (
    _progress_bar, _build_viewer_scene, _build_unity_viewer_html,
)

# ── mm2a pipeline components ────────────────────────────────────────────
from ssv2a.model.pipeline_mm2a_SAM import (
    gemini_filter_local_imgs,
    sam_segment_kept_crops,
    blip_caption_crops,
    clip_embed_texts_l14,
)

# ── AudioLDM 
from ssv2a.model.aldm import build_audioldm, emb_to_audio

# ── CLAP 
from ssv2a.model.clap import clap_embed_auds

# ── Defaults 
DEFAULT_CFG  = "weights/ssv2a.json"
DEFAULT_CKPT = "weights/ssv2a.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_sessions: dict = {}
_depth_model_cache: dict = {}  # {model_name: (model, processor)} — loaded once, reused



# Visual helpers

COLORS = [
    "#FF4444", "#44FF44", "#4488FF", "#FFAA00",
    "#FF44FF", "#00DDDD", "#FFFF44", "#AA44FF",
    "#FF8844", "#44FFAA",
]


def _get_font(size=16):
    for p in [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _draw_bboxes(img, bboxes, labels, statuses=None):
    """Draw bounding boxes on image. statuses: list of 'ON'/'OFF'"""
    draw = ImageDraw.Draw(img)
    font = _get_font(15)
    for i, (box, label) in enumerate(zip(bboxes, labels)):
        color = COLORS[i % len(COLORS)]
        if statuses and statuses[i] == "OFF":
            color = "#666666"
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tag = f"[{i}] {label}"
        if statuses:
            tag += f" ({statuses[i]})"
        draw.text((x1 + 4, y1 + 4), tag, fill=color, font=font)
    return img


def _locality_table_md(labels, localities, gains, pans):
    """Markdown table for locality/gain/pan display."""
    lines = ["| # | Label | Locality | Gain | Gain (dB) | Pan |",
             "|---|-------|----------|------|-----------|-----|"]
    for i, lbl in enumerate(labels):
        g = gains[i]
        db = 20 * math.log10(max(g, 1e-8))
        lines.append("| %d | %s | %.3f | %.3f | %+.1f dB | %+.2f |"
                      % (i, lbl, localities[i], g, db, pans[i]))
    return "\n".join(lines)


def _locality_stereo_mix(mono_waves, localities, pans, sr=16000):
    """Locality-based stereo mix: bigger bbox = louder."""
    n = mono_waves.shape[0]
    samples = mono_waves.shape[-1]
    max_loc = max(localities) if max(localities) > 0 else 1e-6
    norm_locs = [loc / max_loc for loc in localities]
    gains = [nl ** 2.0 for nl in norm_locs]

    stereo = np.zeros((2, samples), dtype=np.float64)
    for i in range(n):
        sig = mono_waves[i, 0].astype(np.float64)
        sig *= gains[i]
        pan = float(np.clip(pans[i], -1.0, 1.0))
        theta = (pan + 1.0) * math.pi / 4.0
        l_gain = math.cos(theta)
        r_gain = math.sin(theta)
        stereo[0] += sig * l_gain
        stereo[1] += sig * r_gain

    peak = max(np.abs(stereo).max(), 1e-8)
    if peak > 0.95:
        stereo *= 0.90 / peak
    return stereo.astype(np.float32)


def _compute_localities_and_pans(bboxes, img_width, img_height):
    """Compute locality (bbox_area / img_area) and pan from bboxes."""
    img_area = img_width * img_height
    localities, pans = [], []
    for box in bboxes:
        area = abs(box[2] - box[0]) * abs(box[3] - box[1])
        localities.append(area / img_area)
        cx = (box[0] + box[2]) / 2.0
        pans.append((cx / img_width - 0.5) * 2.0)
    return localities, pans


def _project_3d_to_localities(objects_3d, cam_pos, cam_fwd, cam_right, cam_up,
                               hfov_deg, img_width, img_height,
                               **_kwargs):
    """Smooth distance-based locality: closer objects = higher locality."""
    half_w = math.tan(math.radians(hfov_deg / 2.0))
    raw_dists, raw_pans, behind = [], [], []

    for obj in objects_3d:
        obj_pos = np.array([obj.X, obj.Y, obj.Z])
        to_obj = obj_pos - cam_pos
        depth = float(np.dot(to_obj, cam_fwd))
        if depth < 0.01:
            raw_dists.append(None); raw_pans.append(0.0); behind.append(True)
            continue
        behind.append(False)
        x_cam = float(np.dot(to_obj, cam_right))
        raw_pans.append(float(np.clip(x_cam / (depth * half_w + 1e-12), -1.0, 1.0)))
        raw_dists.append(max(float(np.linalg.norm(to_obj)), 1e-6))

    valid_dists = [d for d in raw_dists if d is not None]
    if not valid_dists:
        return [0.0] * len(objects_3d), [0.0] * len(objects_3d)

    min_d = min(valid_dists)
    obj_positions = np.array([[o.X, o.Y, o.Z] for o in objects_3d])
    scene_diam = float(np.linalg.norm(obj_positions.max(axis=0) - obj_positions.min(axis=0)))
    scene_diam = max(scene_diam, 1.0)
    CONTRAST = 3.0

    localities = []
    for i in range(len(objects_3d)):
        if behind[i]:
            localities.append(0.0)
        else:
            excess = (raw_dists[i] - min_d) / scene_diam
            localities.append(float(np.clip(1.0 / (1.0 + CONTRAST * excess), 0.05, 1.0)))
    return localities, raw_pans


def _orbit_to_cam_pos(theta_deg: float, phi_deg: float, radius: float,
                      objects_3d) -> np.ndarray:
    """Convert model-viewer orbit params to a 3D camera position."""
    positions = np.array([[o.X, o.Y, o.Z] for o in objects_3d])
    centroid  = positions.mean(axis=0)
    theta_rad = math.radians(theta_deg)
    phi_rad   = math.radians(phi_deg)
    return np.array([
        centroid[0] + radius * math.sin(phi_rad) * math.sin(theta_rad),
        centroid[1] + radius * math.cos(phi_rad),
        centroid[2] + radius * math.sin(phi_rad) * math.cos(theta_rad),
    ])


def step_apply_unity_camera(sid, theta_str, phi_str, radius_str):
    """Called when user clicks 'Use This Camera' in the 3D viewer.
    Converts orbit params → 3D listener position → stores in session.
    """
    if not sid or sid not in _sessions:
        return "No session."
    sess = _sessions[sid]
    objects_3d = sess.get("objects_3d", [])
    if not objects_3d:
        return "⚠ No 3D scene yet — run Step 3 first."
    try:
        theta_deg = float(theta_str) if theta_str not in (None, "", "0") else 0.0
        phi_deg   = float(phi_str)   if phi_str   not in (None, "", "0") else 90.0
        radius    = float(radius_str) if radius_str not in (None, "", "0") else 20.0
    except (ValueError, TypeError):
        return "⚠ Invalid camera params."
    cam_pos = _orbit_to_cam_pos(theta_deg, phi_deg, radius, objects_3d)
    sess["listener_pos"] = cam_pos.tolist()
    sess["listener_theta"] = theta_deg
    print(f"[APPLY CAM] θ={theta_deg:.1f}° φ={phi_deg:.1f}° r={radius:.1f} "
          f"→ pos=({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f})")
    return f"✅ Camera set (θ={theta_deg:.0f}°, φ={phi_deg:.0f}°, r={radius:.1f})"


def _cam_pos_from_session(sess, objects_3d, n, hfov, ww, hh):
    """Return (localities, pans, source_msg) using sess['listener_pos'] if available."""
    listener_pos = sess.get("listener_pos", None)
    if objects_3d and listener_pos and len(objects_3d) >= n:
        cam_pos = np.array(listener_pos)
        positions = np.array([[o.X, o.Y, o.Z] for o in objects_3d])
        centroid  = positions.mean(axis=0)
        cam_fwd   = centroid - cam_pos
        cam_fwd  /= (np.linalg.norm(cam_fwd) + 1e-12)
        world_up  = np.array([0.0, 1.0, 0.0])
        cam_right = np.cross(cam_fwd, world_up)
        r_len = np.linalg.norm(cam_right)
        cam_right = cam_right / r_len if r_len > 1e-6 else np.array([1.0, 0.0, 0.0])
        cam_up    = np.cross(cam_right, cam_fwd)
        localities, pans = _project_3d_to_localities(
            objects_3d[:n], cam_pos, cam_fwd, cam_right, cam_up,
            float(hfov), ww, hh,
        )
        theta = sess.get("listener_theta", 0.0)
        return localities, pans, f"3D listener pos (θ={theta:.0f}°)"
    return None, None, None



# Step 1 — YOLO Detection


def step_yolo_detect(image, hfov):
    """Run YOLO detection + depth estimation on uploaded image."""
    _empty_cb = gr.CheckboxGroup.update(choices=[])
    if image is None:
        yield None, None, "Upload an image first.", "", _empty_cb
        return

    # ── Progress yield 1: starting ───────────────────────────────────
    yield None, None, "⏳ Saving image and loading config…", "", _empty_cb

    # Create session
    sid = f"pipe_{int(time.time())}_{id(image) % 10000}"
    tmp_dir = Path(tempfile.mkdtemp(prefix="3dworld_"))

    # Save uploaded image
    image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    img_path = str(tmp_dir / "input.png")
    image_pil.save(img_path)
    ww, hh = image_pil.size

    # Load config
    cfg_path = DEFAULT_CFG
    if not os.path.exists(cfg_path):
        cfg_path = str(Path().resolve() / "configs" / "model.json")
    with open(cfg_path) as f:
        config_dict = json.load(f)

    # ── Progress yield 2: YOLO running ───────────────────────────────
    yield None, None, "⏳ Running YOLO object detection…", "", _empty_cb

    # YOLO detection (4-tuple: crop_path, locality, label, bbox)
    print(f"\n{'='*60}")
    print(f"  STEP 1 — YOLO Detection")
    print(f"{'='*60}")
    local_imgs = detect_yolo_4tuple(
        [img_path], config_dict["detector"],
        save_dir=str(tmp_dir / "yolo_crops"),
        batch_size=1, device=DEVICE,
    )

    detections = local_imgs.get(img_path, [])
    if not detections:
        yield np.array(image_pil), None, "No objects detected.", sid, _empty_cb
        return

    # ── Progress yield 3: depth estimation ──────────────────────────
    n_det = len(detections)
    yield None, None, f"⏳ Detected {n_det} object(s). Running depth estimation…", "", _empty_cb

    # Depth estimation — use cached model if available
    depth_model_name = config_dict.get("depth_model", "depth-anything/Depth-Anything-V2-Base-hf")
    if depth_model_name not in _depth_model_cache:
        print(f"  [DEPTH] Loading depth model {depth_model_name} (first time)…")
        _depth_model_cache[depth_model_name] = load_depth_model(depth_model_name, DEVICE)
    else:
        print(f"  [DEPTH] Using cached depth model.")
    d_model, d_proc = _depth_model_cache[depth_model_name]
    depth_map = estimate_depth(img_path, d_model, d_proc, DEVICE, model_name=depth_model_name)
    torch.cuda.empty_cache()

    bboxes = [d[3] for d in detections]
    labels = [d[2] for d in detections]
    crop_paths = [d[0] for d in detections]
    localities_raw = [d[1] for d in detections]
    obj_depths = depth_for_bboxes(depth_map, bboxes, method="median")

    # Store session
    _sessions[sid] = dict(
        img_path=img_path, tmp_dir=tmp_dir, config_dict=config_dict,
        image_pil=image_pil.copy(),
        bboxes=bboxes, labels=labels, crop_paths=crop_paths,
        localities_raw=localities_raw, local_imgs=local_imgs,
        img_width=ww, img_height=hh,
        depth_map=depth_map, obj_depths=obj_depths,
        hfov=hfov, detections_4tuple=detections,
    )

    # Annotated images
    annotated = _draw_bboxes(image_pil.copy(), bboxes, labels)
    from ssv2a.data.depth_estimation import save_depth_visualisation
    dv_path = str(tmp_dir / "depth_vis.png")
    save_depth_visualisation(depth_map, dv_path)
    depth_img = Image.open(dv_path)

    # Info
    info_lines = [f"### Detected {len(detections)} object(s)\n"]
    for i, (lbl, box, d) in enumerate(zip(labels, bboxes, obj_depths)):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        loc = localities_raw[i]
        info_lines.append(
            f"- **[{i}] {lbl}** — pixel ({cx:.0f}, {cy:.0f}), depth={d:.1f}, locality={loc:.3f}"
        )

    # Checkbox choices
    choices = [f"[{i}] {lbl}" for i, lbl in enumerate(labels)]

    yield (
        np.array(annotated),
        np.array(depth_img),
        "\n".join(info_lines),
        sid,
        gr.CheckboxGroup.update(choices=choices, value=choices),  # all ON by default
    )



# Step 1b — Apply object toggle (redraw with ON/OFF)


def step_toggle_objects(sid, selected_objects):
    """Update the display to show which objects are ON vs OFF."""
    if not sid or sid not in _sessions:
        return None, "No session."

    sess = _sessions[sid]
    bboxes = sess["bboxes"]
    labels = sess["labels"]
    n = len(labels)

    # Parse selected
    selected_indices = set()
    for s in (selected_objects or []):
        # Format: "[0] label"
        try:
            idx = int(s.split("]")[0].strip("["))
            selected_indices.add(idx)
        except:
            pass

    statuses = ["ON" if i in selected_indices else "OFF" for i in range(n)]
    sess["object_statuses"] = statuses

    annotated = _draw_bboxes(sess["image_pil"].copy(), bboxes, labels, statuses)

    on_count = statuses.count("ON")
    off_count = statuses.count("OFF")
    info = f"**{on_count} ON** / {off_count} OFF — {n} total objects"
    kept = [f"  ✅ [{i}] {labels[i]}" for i in range(n) if statuses[i] == "ON"]
    dropped = [f"  ❌ [{i}] {labels[i]}" for i in range(n) if statuses[i] == "OFF"]
    info += "\n" + "\n".join(kept + dropped)

    return np.array(annotated), info



# Step 2 — Auto-chain: LLM Filter → BLIP → CLIP Embeddings + SAM (3D only)


def step_auto_chain(sid, skip_llm):
    """Run the full preprocessing chain on user-kept objects.

    Steps:
    1. Filter to user-selected objects (ON/OFF toggles)
    2. LLM filter (GPT-4o) — drop silent objects (unless skipped)
    3. SAM segmentation — isolate each kept object (used for 3D world only)
    4. BLIP captioning — caption each original (pre-SAM) filtered crop
    5. CLIP embedding — image CLIP (original crop) + text CLIP (BLIP caption)

    Returns intermediate results for display.
    """
    if not sid or sid not in _sessions:
        yield None, None, None, "No session. Run Step 1 first."
        return

    sess = _sessions[sid]
    img_path = sess["img_path"]
    bboxes = sess["bboxes"]
    labels = sess["labels"]
    statuses = sess.get("object_statuses", ["ON"] * len(labels))
    detections = sess["detections_4tuple"]

    log = ["## Step 2 — Processing Pipeline\n"]

    # ── 2a. User toggle filter ──────────────────────────────────────────
    kept_indices = [i for i, s in enumerate(statuses) if s == "ON"]
    if not kept_indices:
        yield None, None, None, "All objects are OFF. Turn some ON first."
        return

    # Build local_imgs dict with only kept objects (4-tuple format)
    kept_detections = [detections[i] for i in kept_indices]
    local_imgs = {img_path: kept_detections}
    kept_labels = [labels[i] for i in kept_indices]

    log.append(f"### 2a. User Selection\n")
    log.append(f"**Kept {len(kept_indices)} / {len(labels)}** objects after user toggle")
    for i in kept_indices:
        log.append(f"  ✅ [{i}] {labels[i]}")
    log.append("")

    # ── 2b. LLM Filter 
    if not skip_llm:
        log.append("### 2b. LLM Sound-Source Filter (GPT-4o)\n")
        log.append("Sending each crop + full scene to GPT-4o Vision …")
        yield None, None, None, "\n\n".join(log)  # keep connection alive
        try:
            local_imgs, filter_report = gemini_filter_local_imgs(local_imgs)
            for entry in filter_report:
                for k in entry.get("kept", []):
                    log.append(f"  ✅ KEPT: {k['name']}")
                for d in entry.get("dropped", []):
                    log.append(f"  ❌ DROPPED: {d['name']}")
        except Exception as e:
            log.append(f"  ⚠ LLM filter failed: {e}")
            log.append("  Continuing with all user-selected objects.")
    else:
        log.append("### 2b. LLM Filter — **SKIPPED** (user chose to skip)\n")

    remaining = local_imgs.get(img_path, [])

    # Build LLM kept/dropped gallery images from original crops
    _remaining_set = {r[0] for r in remaining}
    kept_gallery_items, dropped_gallery_items = [], []
    for _d in kept_detections:
        try:
            _img = np.array(Image.open(_d[0]).convert("RGB"))
            if _d[0] in _remaining_set:
                kept_gallery_items.append((_img, _d[2]))
            else:
                dropped_gallery_items.append((_img, _d[2]))
        except Exception:
            pass

    if not remaining:
        yield kept_gallery_items, dropped_gallery_items, None, "\n\n".join(log) + "\n\n**All objects dropped by LLM filter!**"
        return

    after_llm_labels = [r[2] for r in remaining]
    after_llm_bboxes = [r[3] for r in remaining]
    log.append(f"\n**{len(remaining)} objects** after LLM filter\n")

    # Save LLM-filtered crops BEFORE SAM — BLIP will use these, not SAM masks
    llm_filtered_local_imgs = {img_path: list(remaining)}

    # ── 2c. SAM Segmentation ───────────────────────────────────────────
    log.append("### 2c. SAM Segmentation\n")
    log.append("Running SAM on each kept object (bbox + center point prompt) …")
    yield kept_gallery_items, dropped_gallery_items, None, "\n\n".join(log)  # keep connection alive
    try:
        sam_dir = str(sess["tmp_dir"] / "sam_crops")
        local_imgs = sam_segment_kept_crops(
            local_imgs, save_dir=sam_dir,
            sam_model="sam_b.pt", device=DEVICE,
        )
        torch.cuda.empty_cache()
        sam_crops = local_imgs.get(img_path, [])
        for sc in sam_crops:
            log.append(f"  🎯 SAM: {Path(sc[0]).name}")
        log.append(f"\n**{len(sam_crops)} SAM-masked crops** produced\n")
    except Exception as e:
        log.append(f"  ⚠ SAM failed: {e}")
        log.append("  Continuing with original crops.")
        sam_crops = remaining

    # Build SAM gallery
    sam_images = []
    for sc in sam_crops:
        try:
            sam_images.append(np.array(Image.open(sc[0]).convert("RGB")))
        except:
            pass

    # Show SAM results immediately
    combined_sam = None
    if sam_images:
        h_target = 256
        resized = []
        for si in sam_images:
            pil_si = Image.fromarray(si)
            ratio = h_target / pil_si.height
            pil_si = pil_si.resize((int(pil_si.width * ratio), h_target))
            resized.append(np.array(pil_si))
        combined_sam = np.concatenate(resized, axis=1)
    yield kept_gallery_items, dropped_gallery_items, combined_sam, "\n\n".join(log)  # show SAM crops while BLIP loads

    # ── 2d. BLIP Captioning ────────────────────────────────────────────
    log.append("### 2d. BLIP Captioning\n")
    log.append("Running BLIP on GPT-4o filtered crops (original YOLO crops, not SAM masks) …")
    try:
        captions = blip_caption_crops(llm_filtered_local_imgs, device=DEVICE)
        caption_list = captions.get(img_path, [])
        for i, cap in enumerate(caption_list):
            lbl = after_llm_labels[i] if i < len(after_llm_labels) else "?"
            log.append(f'  📝 [{i}] {lbl}: **"{cap}"**')
        log.append("")
        sess["captions"] = caption_list
    except Exception as e:
        log.append(f"  ⚠ BLIP failed: {e}")
        caption_list = [f"the sound of {lbl}" for lbl in after_llm_labels]
        sess["captions"] = caption_list

    # Unload BLIP before loading CLIP to free GPU memory
    try:
        unload_blip()
    except Exception:
        pass
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # ── 2e. CLIP Embeddings ────────────────────────────────────────────
    log.append("### 2e. CLIP/CLAP Embeddings\n")
    log.append("Computing CLIP ViT-L/14 image + text embeddings …")
    yield kept_gallery_items, dropped_gallery_items, combined_sam, "\n\n".join(log)  # keep connection alive
    try:
        # Image CLIP embeddings — use original GPT-4o filtered crops, not SAM masks
        crop_paths_final = [r[0] for r in llm_filtered_local_imgs[img_path]]
        img_clip_embeds = clip_embed_images(crop_paths_final, device=DEVICE)

        # Text CLIP embeddings (from BLIP captions)
        text_clip_embeds = clip_embed_texts_l14(caption_list, device=DEVICE)

        n_obj = len(crop_paths_final)
        log.append(f"  🖼  Image CLIP: {n_obj} × {img_clip_embeds.shape[-1]}-dim")
        log.append(f"  📝 Text CLIP:  {n_obj} × {text_clip_embeds.shape[-1]}-dim")

        # Cosine similarity between image and text per object
        img_norm = img_clip_embeds / (img_clip_embeds.norm(dim=-1, keepdim=True) + 1e-8)
        txt_norm = text_clip_embeds / (text_clip_embeds.norm(dim=-1, keepdim=True) + 1e-8)
        cos_sims = (img_norm * txt_norm).sum(dim=-1)
        for i in range(n_obj):
            lbl = after_llm_labels[i] if i < len(after_llm_labels) else "?"
            log.append(f"    [{i}] {lbl}: img↔text cosine = {cos_sims[i]:.4f}")

        sess["img_clip_embeds"] = img_clip_embeds
        sess["text_clip_embeds"] = text_clip_embeds
        log.append(f"\n**Embeddings ready** — {n_obj} objects × {img_clip_embeds.shape[-1]}-dim\n")
    except Exception as e:
        log.append(f"  ⚠ Embedding failed: {e}")

    # ── Update session 
    sess["local_imgs_filtered"] = local_imgs
    sess["after_llm_labels"] = after_llm_labels
    sess["after_llm_bboxes"] = after_llm_bboxes
    sess["sam_crop_paths"] = [sc[0] for sc in sam_crops]          # SAM → 3D world only
    sess["llm_crop_paths"] = [r[0] for r in llm_filtered_local_imgs[img_path]]  # original → audio CLIP

    yield kept_gallery_items, dropped_gallery_items, combined_sam, "\n\n".join(log)



# Shared helper: get filtered object set


def _get_active_objects(sess):
    """Return (labels, bboxes, crop_paths) for objects that survived ALL filters.

    Priority:
      1. If Step 2 ran → use LLM-filtered + SAM results (already reflects toggle)
      2. Else → apply toggle ON/OFF from Step 1 to raw YOLO detections
    """
    if "after_llm_labels" in sess:
        return (
            list(sess["after_llm_labels"]),
            list(sess["after_llm_bboxes"]),
            list(sess.get("llm_crop_paths", sess.get("sam_crop_paths", sess["crop_paths"]))),
        )

    # Fall back to toggle filter
    labels = sess["labels"]
    bboxes = sess["bboxes"]
    crop_paths = sess["crop_paths"]
    statuses = sess.get("object_statuses", ["ON"] * len(labels))

    active_labels, active_bboxes, active_crops = [], [], []
    for i, s in enumerate(statuses):
        if s == "ON":
            active_labels.append(labels[i])
            active_bboxes.append(bboxes[i])
            active_crops.append(crop_paths[i])

    return active_labels, active_bboxes, active_crops



# Step 3 — Generate 3D World


def step_generate_3d(sid, mesh_resolution, hfov):
    """SAM segment → Hunyuan3D-2 mesh → 3D world. Mirror of gradio_3d_world.step_generate_3d_world."""
    if not sid or sid not in _sessions:
        yield None, None, "No session.", ""
        return

    sess = _sessions[sid]
    hfov = float(hfov)
    resolution = int(mesh_resolution)

    # ── Use only objects that survived ALL filters (toggle + LLM) ──────
    labels, bboxes, crop_paths = _get_active_objects(sess)
    if not labels:
        yield None, None, "⚠ No active objects — turn some ON in Step 1 or re-run Step 2.", ""
        return

    ww, hh = sess["img_width"], sess["img_height"]
    image_pil = sess["image_pil"]
    depth_map = sess["depth_map"]
    obj_depths = depth_for_bboxes(depth_map, bboxes, method="median")

    cam = CameraIntrinsics.from_image_size(ww, hh, hfov_deg=hfov)
    objects_3d = backproject_objects(
        bboxes, labels, obj_depths, ww, hh, camera=cam, hfov_deg=hfov,
    )

    n_obj = len(objects_3d)
    log_lines = [f"### 3D World — {n_obj} objects (SAM + Hunyuan3D-2)\n"]
    total_t = time.time()

    print(f"\n{'='*60}")
    print(f"  STEP 3 — GENERATE 3D WORLD  ({n_obj} objects)")
    print(f"{'='*60}")

    # ── Phase 1: SAM-segment every object from the full image ──────────
    print(f"\n  Phase 1/5: SAM Segmentation")
    log_lines.append("**Phase 1/5: SAM Segmentation …**")
    yield None, None, "\n".join(log_lines), ""  # keep alive

    isolated_crops = []
    for i, (obj, bbox) in enumerate(zip(objects_3d, bboxes)):
        _progress_bar(i, n_obj, f"SAM [{i}] {obj.label}")
        isolated = segment_object(image_pil, bbox, device=DEVICE)
        print(f"    [{i}] SAM-isolated '{obj.label}' ({isolated.size[0]}x{isolated.size[1]})")
        isolated_crops.append(isolated)
        log_lines.append(f"  ✂ SAM [{i}] {obj.label}")
        yield None, None, "\n".join(log_lines), ""  # progress after each object
    _progress_bar(n_obj, n_obj, "SAM done")
    unload_sam()
    torch.cuda.empty_cache()

    # ── Phase 2: Hunyuan3D-2 mesh per object ───────────────────────────
    print(f"\n  Phase 2/5: Hunyuan3D-2 Mesh Generation")
    log_lines.append("\n**Phase 2/5: Hunyuan3D-2 Mesh Generation …**")
    yield None, None, "\n".join(log_lines), ""  # keep alive before first mesh

    meshes = []
    for i, (obj, crop) in enumerate(zip(objects_3d, isolated_crops)):
        _progress_bar(i, n_obj, f"Generating [{i}] {obj.label}")
        print(f"    [{i}] Starting Hunyuan3D-2 for '{obj.label}' ...")
        log_lines.append(f"  ⏳ Generating mesh [{i}] {obj.label} (this takes 2-5 min)…")
        yield None, None, "\n".join(log_lines), ""  # keep alive while mesh generates
        t0 = time.time()
        mesh = generate_object_mesh(
            obj.label, image_crop=crop, device=DEVICE, resolution=resolution,
        )
        dt = time.time() - t0
        meshes.append(mesh)
        print(f"    [{i}] ✓ {obj.label}: {len(mesh.vertices):,} verts, "
              f"{len(mesh.faces):,} faces ({dt:.1f}s)")
        log_lines.append(f"  ✅ **[{i}] {obj.label}** — mesh ready")
        yield None, None, "\n".join(log_lines), ""  # confirm mesh done
    _progress_bar(n_obj, n_obj, "All meshes generated")
    torch.cuda.empty_cache()

    # ── Phase 3: Assemble scene ─────────────────────────────────────────
    print(f"\n  Phase 3/5: Assembling 3D World")
    log_lines.append("\n**Phase 3/5: Assembling 3D scene …**")
    yield None, None, "\n".join(log_lines), ""

    max_z = max(o.Z for o in objects_3d) if objects_3d else 50
    max_x = max(abs(o.X) for o in objects_3d) if objects_3d else 30
    platform = create_grid_platform(
        extent_x=max(max_x * 2, 60), extent_z=max(max_z * 1.5, 80),
    )
    scene = build_3d_world(
        meshes, objects_3d, camera_fx=cam.fx, camera_fy=cam.fy,
        platform=platform, boost_colors=True,
    )
    out_dir = str(sess["tmp_dir"])
    scene_path = os.path.join(out_dir, "3d_world.glb")
    export_scene_glb(scene, scene_path)
    full_mb = os.path.getsize(scene_path) / (1024 * 1024)
    print(f"    ✓ Full scene GLB: {full_mb:.1f} MB")

    for m in meshes:
        _boost_vertex_colors(m, brightness=1.4, saturation=1.15)
    indiv_paths = export_individual_glbs(
        meshes, labels, os.path.join(out_dir, "meshes"),
    )
    first_mesh_path = indiv_paths[0] if indiv_paths else None

    # ── Phase 4: Export scene.json ──────────────────────────────────────
    print(f"\n  Phase 4/5: Export Metadata")
    scene_json_path = os.path.join(out_dir, "scene.json")
    export_scene_json(
        objects_3d, meshes, scene_path, indiv_paths, scene_json_path,
        camera_fx=cam.fx, camera_fy=cam.fy,
    )
    sess["scene_json_path"] = scene_json_path
    sess["scene_dir"] = out_dir
    print(f"    ✓ scene.json written")

    # ── Phase 5: Decimated viewer GLB ──────────────────────────────────
    print(f"\n  Phase 5/5: Decimated Viewer GLB")
    log_lines.append("\n**Phase 5/5: Building viewer …**")
    yield None, None, "\n".join(log_lines), ""

    viewer_glb_path = os.path.join(out_dir, "viewer.glb")
    try:
        t_dec = time.time()
        viewer_scene = _build_viewer_scene(meshes, objects_3d, cam.fx, cam.fy)
        export_scene_glb(viewer_scene, viewer_glb_path)
        viewer_kb = os.path.getsize(viewer_glb_path) // 1024
        print(f"  [VIEWER] ✓ viewer.glb: {viewer_kb:,} KB ({time.time()-t_dec:.1f}s)")
    except Exception as _ve:
        print(f"  [VIEWER] ✗ Decimation failed: {_ve} — falling back to full GLB")
        viewer_glb_path = scene_path

    # Build interactive Three.js viewer HTML
    unity_html = _build_unity_viewer_html(
        viewer_glb_path, objects_3d, meshes,
        cam.fx, cam.fy, img_width=ww, img_height=hh,
    )

    total_dt = time.time() - total_t
    log_lines.append(f"\n**Done — {len(meshes)} objects generated in {total_dt:.0f}s ✓**")

    sess["meshes"] = meshes
    sess["objects_3d"] = objects_3d
    sess["scene_path"] = scene_path
    sess["indiv_mesh_paths"] = indiv_paths

    yield scene_path, first_mesh_path, "\n".join(log_lines), unity_html



# Step 4 — Generate Spatial Audio


# ── Subprocess helper for audio (isolates CUDA context from Hunyuan3D) ────
import subprocess as _sp
import tempfile as _tf
import json as _json

_AUDIO_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_worker.py")
_AUDIO_PYTHON = sys.executable  # same ssv2a_env Python


def _run_audio_worker(images_arg, save_dir: str, params: dict) -> None:
    """
    Run audio_worker.py in a fresh subprocess so it gets its own CUDA context,
    completely isolated from the context that Hunyuan3D corrupted via os._exit.
    images_arg: list of paths (reference) or dict {path: [(path, weight)]} (per-object)
    """
    with _tf.TemporaryDirectory() as td:
        params_file = os.path.join(td, "params.json")
        result_file = os.path.join(td, "result.json")

        # Tuples are not JSON-serialisable — convert to lists
        if isinstance(images_arg, dict):
            images_serial = {k: [list(v) for v in vlist]
                             for k, vlist in images_arg.items()}
        else:
            images_serial = [str(p) for p in images_arg]

        payload = dict(params)
        payload["images"]   = images_serial
        payload["save_dir"] = save_dir

        with open(params_file, "w") as fh:
            _json.dump(payload, fh)

        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

        proc = _sp.run(
            [_AUDIO_PYTHON, _AUDIO_WORKER,
             "--params-json", params_file,
             "--result-json", result_file],
            env=env,
        )
        if proc.returncode not in (0, None):
            raise RuntimeError(
                f"audio_worker failed (rc={proc.returncode}). "
                "Check terminal output above for the full traceback."
            )
        if not os.path.exists(result_file):
            raise RuntimeError(
                "audio_worker finished but wrote no result.json — check stderr above."
            )
        with open(result_file) as fh:
            res = _json.load(fh)
        if res.get("status") != "ok":
            raise RuntimeError(f"audio_worker returned status={res.get('status')}")


def step_generate_audio(sid,
                        batch_size, var_samples, cycle_its, cycle_samples,
                        duration, seed, hfov):
    """Generate per-object audio + locality stereo mix."""
    import soundfile as _sf

    if not sid or sid not in _sessions:
        return None, None, "No session."

    sess = _sessions[sid]
    set_seed(int(seed))

    img_path = sess["img_path"]
    ww, hh = sess["img_width"], sess["img_height"]

    # ── Use only objects that survived ALL filters (toggle + LLM) ──────
    labels, bboxes, crop_paths = _get_active_objects(sess)
    if not labels:
        return None, None, "⚠ No active objects — turn some ON in Step 1 or re-run Step 2."

    bs = int(batch_size); vs = int(var_samples)
    ci = int(cycle_its); cs = int(cycle_samples)
    dur = int(duration)
    n = len(crop_paths)

    log = [f"## Step 4 — Audio Generation ({n} objects)\n"]

    # Shared params passed to the audio subprocess
    _base_params = dict(
        text="", transcription="",
        config=DEFAULT_CFG, pretrained=DEFAULT_CKPT,
        gen_remix=True, gen_tracks=False, emb_only=False,
        shuffle_remix=True,
        batch_size=bs, var_samples=vs,
        cycle_its=ci, cycle_samples=cs,
        duration=dur, seed=int(seed), device=DEVICE,
    )

    # ── Reference audio ────────────────────────────────────────────────
    log.append("[1/2] Reference audio (full image) …")
    ref_dir = sess["tmp_dir"] / "ref_audio"
    ref_dir.mkdir(exist_ok=True)
    _run_audio_worker(
        [str(img_path)],
        save_dir=str(ref_dir),
        params=dict(_base_params, keep_data_cache=False),
    )

    ref_wav_path = str(ref_dir / (Path(img_path).stem + ".wav"))
    ref_wave, ref_sr = _sf.read(ref_wav_path)
    if ref_wave.ndim > 1:
        ref_wave = ref_wave[:, 0]
    ref_wave = np.array(ref_wave, dtype=np.float32)
    log.append(f"  Reference done. Peak={float(np.abs(ref_wave).max()):.4f}")

    # ── Per-object audio ───────────────────────────────────────────────
    log.append(f"[2/2] Per-object audio ({n} objects) …")
    tracks_dir = sess["tmp_dir"] / "tracks"
    tracks_dir.mkdir(exist_ok=True)

    per_object_waves = []
    captions = sess.get("captions", [])
    use_blip = len(captions) == n

    for i, (cp, lbl) in enumerate(zip(crop_paths, labels)):
        log.append(f"  [{i+1}/{n}] {lbl} …")
        obj_dir = tracks_dir / f"obj_{i}"
        obj_dir.mkdir(exist_ok=True)

        if use_blip:
            caption = captions[i]
            log.append(f"    BLIP caption: \"{caption}\" → combined image+text CLIP")
            obj_wav = str(obj_dir / "audio.wav")
            _run_audio_worker(
                [cp],
                save_dir=obj_wav,
                params=dict(_base_params, caption=caption),
            )
        else:
            crop_dict = {cp: [(cp, 1.0)]}
            _run_audio_worker(
                crop_dict,
                save_dir=str(obj_dir),
                params=dict(_base_params, keep_data_cache=True),
            )
            obj_wav = str(obj_dir / (Path(cp).stem + ".wav"))
        wave, _ = _sf.read(obj_wav)
        if wave.ndim > 1:
            wave = wave[:, 0]
        per_object_waves.append(np.array(wave, dtype=np.float32))
        log.append(f"    Peak={float(np.abs(wave).max()):.4f}")

    # Stack
    max_len = max(w.shape[0] for w in per_object_waves)
    waves_arr = np.zeros((n, 1, max_len), dtype=np.float32)
    for i, w in enumerate(per_object_waves):
        waves_arr[i, 0, :w.shape[0]] = w

    # ── Locality + pan 
    objects_3d = sess.get("objects_3d", [])

    localities, pans, source_msg = _cam_pos_from_session(sess, objects_3d, n, hfov, ww, hh)
    if localities is not None:
        log.append(source_msg)
    else:
        localities, pans = _compute_localities_and_pans(bboxes, ww, hh)
        log.append("Using YOLO bbox localities")

    _max_loc = max(localities) if max(localities) > 0 else 1e-6
    gains = [(loc / _max_loc) ** 2.0 for loc in localities]

    for i in range(n):
        log.append(f"  {labels[i]}: locality={localities[i]:.3f}  gain={gains[i]:.3f}  pan={pans[i]:+.2f}")

    # ── Cache 
    sess["per_object_waves"] = waves_arr
    sess["audio_labels"] = list(labels)
    sess["localities"] = localities
    sess["pans"] = pans
    sess["ref_wave"] = ref_wave
    sess["ref_sr"] = ref_sr

    # ── Mix 
    stereo = _locality_stereo_mix(waves_arr, localities, pans, sr=SAMPLE_RATE)
    stereo_out = stereo.T.astype(np.float32)

    table = _locality_table_md(labels, localities, gains, pans)
    status = f"Done ✓ ({stereo_out.shape[0] / SAMPLE_RATE:.1f}s @ {SAMPLE_RATE} Hz)\n\n" + table

    return (SAMPLE_RATE, stereo_out), (ref_sr, ref_wave), status


# Step 4b — Re-render (camera move)


def step_rerender(sid, hfov):
    """Re-mix per-object tracks from camera position stored in session."""
    if not sid or sid not in _sessions:
        return None, None, "No session."
    sess = _sessions[sid]
    if "per_object_waves" not in sess:
        return None, None, "Generate audio first."

    waves = sess["per_object_waves"]
    labels = sess["audio_labels"]
    ref_wave = sess["ref_wave"]
    ref_sr = sess["ref_sr"]
    ww, hh = sess["img_width"], sess["img_height"]
    n = waves.shape[0]

    objects_3d = sess.get("objects_3d", [])
    log = [f"Re-rendering ({n} objects) …"]

    localities, pans, source_msg = _cam_pos_from_session(sess, objects_3d, n, hfov, ww, hh)
    if localities is None:
        localities = sess.get("localities", [0.1] * n)
        pans = sess.get("pans", [0.0] * n)
        source_msg = "Using stored localities"
    log.append(source_msg)

    _max_loc = max(localities) if max(localities) > 0 else 1e-6
    gains = [(loc / _max_loc) ** 2.0 for loc in localities]

    stereo = _locality_stereo_mix(waves, localities, pans, sr=SAMPLE_RATE)
    stereo_out = stereo.T.astype(np.float32)

    for i in range(n):
        log.append(f"  {labels[i]}: locality={localities[i]:.3f}  gain={gains[i]:.3f}  pan={pans[i]:+.2f}")

    table = _locality_table_md(labels, localities, gains, pans)
    return (SAMPLE_RATE, stereo_out), (ref_sr, ref_wave), "Done ✓ (instant)\n\n" + table



# Build Gradio UI


def build_ui():
    with gr.Blocks(
        title="SSV2A Full Pipeline",
        theme=gr.themes.Soft(),
        css="""
        .step-header { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 22px; font-weight: 700; }
        #apply-cam-btn-row { display: none !important; }
        """
    ) as demo:

        gr.Markdown(
            "# 🎵 SSV2A — Full Pipeline\n"
            "**Image → YOLO → ON/OFF Toggle → LLM Filter → SAM → BLIP → CLIP Embeddings → 3D World → Spatial Audio**\n\n"
            "Upload an image and walk through each pipeline step."
        )

        session_id = gr.State(value="")

        # Step 1: Upload + YOLO
        gr.Markdown("---\n## Step 1 — Upload Image & YOLO Detection")

        with gr.Row():
            image_input = gr.Image(label="Upload Image", type="numpy")
        hfov_slider = gr.State(60)  # fixed at 60 degrees

        detect_btn = gr.Button("① Detect Objects + Depth", variant="primary")

        with gr.Row():
            annotated_img = gr.Image(label="Detected Objects", interactive=False)
            depth_img = gr.Image(label="Depth Map", interactive=False)

        detect_info = gr.Markdown("Upload an image and click Detect.")

        # Object toggle
        gr.Markdown("### Toggle Objects ON/OFF")
        gr.Markdown("Uncheck objects you want to exclude from the pipeline.")
        object_checkboxes = gr.CheckboxGroup([], label="Objects", interactive=True)
        toggle_info = gr.Markdown("")

        
        # Step 2: Auto-chain (LLM → SAM → BLIP → Embeddings)
    
        gr.Markdown("---\n## Step 2 — LLM Filter → BLIP → CLIP Embeddings + SAM (3D only)")
        gr.Markdown(
            "This step auto-chains: **LLM sound-source filter** (GPT-4o) → "
            "**BLIP captioning** (on original filtered crops) → "
            "**CLIP embeddings** (image + text) → **SAM segmentation** (for 3D world only).\n\n"
            "Audio generation uses the original YOLO crops + BLIP captions. SAM masks are only used in Step 3 for 3D mesh generation."
        )

        skip_llm = gr.Checkbox(label="Skip LLM filter (keep all objects)", value=False)
        chain_btn = gr.Button("② Run Processing Chain", variant="primary")

        with gr.Row():
            llm_kept_gallery = gr.Gallery(label="✅ Kept after LLM filter", columns=4, height=200, interactive=False)
            llm_dropped_gallery = gr.Gallery(label="❌ Dropped by LLM filter", columns=4, height=200, interactive=False)
        sam_gallery = gr.Image(label="🎯 SAM-Segmented Objects", interactive=False)
        chain_log = gr.Markdown("Click ② to run the processing chain.")

        
        # Step 3: 3D World
        
        gr.Markdown("---\n## Step 3 — Generate 3D World")
        gr.Markdown(
            "Segments each object with **SAM**, generates a 3D mesh using **Hunyuan3D-2**, "
            "and positions each mesh in 3D space using **depth estimation + camera intrinsics**.\n\n"
            "⚠ This step takes **2-5 minutes** per object."
        )

        mesh_res_slider = gr.State(256)  # Mesh Resolution fixed at 256
        gen_3d_btn = gr.Button("③ Generate 3D World", variant="primary")
        world_html = gr.HTML(label="Interactive 3D Viewer")
        gen_3d_info = gr.Markdown("")

        
        # Step 4: Spatial Audio
    
        gr.Markdown("---\n## Step 4 — Generate Spatial Audio")

        bs_sl = gr.State(64)   # Batch Size fixed
        vs_sl = gr.State(64)   # Var Samples fixed
        ci_sl = gr.State(64)   # Cycle Iterations fixed
        cs_sl = gr.State(64)   # Cycle Samples fixed
        seed_input = gr.State(42)  # Seed fixed
        dur_sl = gr.State(10)  # Duration fixed at 10s

        gen_audio_btn = gr.Button("④ Generate Spatial Audio", variant="primary")

        with gr.Row():
            spatial_audio = gr.Audio(label="3D Spatial Stereo Mix", type="numpy")
            mono_audio = gr.Audio(label="Mono Reference", type="numpy")

        gen_audio_info = gr.Markdown("")

        # ── Re-render controls ──
        gr.Markdown("### Re-render from Camera Angle")
        gr.Markdown(
            "Orbit the 3D viewer above to your desired position, "
            "then click **📍 Use This Camera** and then **🔄 Re-render (instant)**."
        )

        # Hidden row targeted by viewer JS (display:none via CSS)
        with gr.Row(elem_id="apply-cam-btn-row"):
            unity_cam_theta  = gr.Number(value=0,  label="theta",  elem_id="unity_cam_x")
            unity_cam_phi    = gr.Number(value=90, label="phi",    elem_id="unity_cam_y")
            unity_cam_radius = gr.Number(value=20, label="radius", elem_id="unity_cam_z")
            apply_cam_btn    = gr.Button("Apply 3D Camera", elem_id="apply-cam-btn")

        apply_cam_status = gr.Markdown("")
        rerender_btn = gr.Button("🔄 Re-render (instant)", variant="secondary")

        
        # Example images
        
        example_dir = Path("examples")
        if example_dir.exists():
            examples = sorted(str(p) for p in example_dir.glob("*.png"))
            examples += sorted(str(p) for p in example_dir.glob("*.jpg"))
            if examples:
                gr.Markdown("---\n### Example Images")
                gr.Examples(examples=examples, inputs=[image_input])

        
        # Wiring
        

        # Step 1: Detect
        detect_btn.click(
            fn=step_yolo_detect,
            inputs=[image_input, hfov_slider],
            outputs=[annotated_img, depth_img, detect_info, session_id, object_checkboxes],
        )

        # Step 1b: Toggle
        object_checkboxes.change(
            fn=step_toggle_objects,
            inputs=[session_id, object_checkboxes],
            outputs=[annotated_img, toggle_info],
        )

        # Step 2: Auto-chain
        chain_btn.click(
            fn=step_auto_chain,
            inputs=[session_id, skip_llm],
            outputs=[llm_kept_gallery, llm_dropped_gallery, sam_gallery, chain_log],
        )

        # Step 3: 3D World
        gen_3d_btn.click(
            fn=step_generate_3d,
            inputs=[session_id, mesh_res_slider, hfov_slider],
            outputs=[gr.State(), gr.State(), gen_3d_info, world_html],
        )

        # Step 4: Audio
        gen_audio_btn.click(
            fn=step_generate_audio,
            inputs=[session_id,
                    bs_sl, vs_sl, ci_sl, cs_sl,
                    dur_sl, seed_input, hfov_slider],
            outputs=[spatial_audio, mono_audio, gen_audio_info],
        )

        # Step 4b: Re-render
        rerender_btn.click(
            fn=step_rerender,
            inputs=[session_id, hfov_slider],
            outputs=[spatial_audio, mono_audio, gen_audio_info],
        )

        # Apply camera from viewer (hidden button auto-clicked by JS)
        apply_cam_btn.click(
            fn=step_apply_unity_camera,
            inputs=[session_id, unity_cam_theta, unity_cam_phi, unity_cam_radius],
            outputs=[apply_cam_status],
        )

    return demo



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SSV2A — Full Pipeline — Gradio UI")
    parser.add_argument("--port", type=int, default=7875)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    # Same FastAPI+uvicorn pattern as gradio_3d_world.py so GLB files are served
    from fastapi import FastAPI as _FastAPI
    from fastapi.responses import HTMLResponse as _HTMLResponse, Response as _Response
    import uvicorn as _uvicorn

    app = _FastAPI()

    @app.get("/viewer_page")
    async def _serve_viewer_page(path: str = ""):
        if (not path.startswith("/tmp/3dworld_")
                or not path.endswith("/viewer.html")
                or ".." in path):
            return _HTMLResponse("Forbidden", status_code=403)
        try:
            with open(path, "r") as fh:
                return _HTMLResponse(fh.read())
        except FileNotFoundError:
            return _HTMLResponse("Not found", status_code=404)

    @app.get("/viewer_asset")
    async def _serve_viewer_asset(path: str = ""):
        if not path.startswith("/tmp/3dworld_") or ".." in path:
            return _Response("Forbidden", status_code=403)
        _allowed_ext = (".glb", ".gltf", ".bin", ".png", ".jpg", ".json")
        if not any(path.lower().endswith(ext) for ext in _allowed_ext):
            return _Response("Forbidden file type", status_code=403)
        import mimetypes
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        try:
            with open(path, "rb") as fh:
                return _Response(fh.read(), media_type=ctype)
        except FileNotFoundError:
            return _Response("Not found", status_code=404)

    demo = build_ui()
    demo.queue(concurrency_count=1)
    app = gr.mount_gradio_app(app, demo, path="/")

    print(f"\n  Launching on http://0.0.0.0:{args.port}\n")
    _uvicorn.run(app, host="0.0.0.0", port=args.port)