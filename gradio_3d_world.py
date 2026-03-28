"""
gradio_3d_world.py — Generative 3D World + Spatial Audio

Extends the existing SSV2A pipeline with generative 3D mesh creation:
  1. Upload an image
  2. YOLO detects objects + monocular depth estimation
  3. BLIP captions each crop → SD Turbo renders clean images → Hunyuan3D-2 generates PBR-textured 3D meshes
  4. Meshes are positioned in 3D world space using depth + camera model
  5. Combined scene displayed as interactive 3D viewer (.glb)
  6. Generate 3D spatial stereo audio from the scene

Launch
------
    python gradio_3d_world.py                     # default port 7860
    python gradio_3d_world.py --port 7861 --share  # public link
"""

from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── numpy 2.x compat: restore aliases removed in numpy 2.0 ────────────────
# Gradio 3.50.2 references np.bool8, np.float_, np.obj2sctype which were
# removed in NumPy 2.0.  Monkey-patch them back so Gradio can function.
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

# ── HF cache: redirect to /Data to avoid home-dir quota ───────────────────
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

# ── SSV2A core ──────────────────────────────────────────────────────────────
from ssv2a.data.detect_spatial import detect as detect_spatial
from ssv2a.data.utils import clip_embed_images, emb2seq, save_wave, set_seed
from ssv2a.model.pipeline import Pipeline, image_to_audio as i2a_image_to_audio
from ssv2a.model.aldm import build_audioldm, emb_to_audio

# ── Depth estimation ────────────────────────────────────────────────────────
from ssv2a.data.depth_estimation import (
    load_depth_model,
    estimate_depth,
    depth_for_bboxes,
)

# ── 3D spatial ──────────────────────────────────────────────────────────────
from ssv2a.model.spatial_3d import (
    CameraIntrinsics,
    Object3D,
    backproject_objects,
    apply_listener_rotation,
    compute_3d_spatial_params,
    SAMPLE_RATE,
)

# ── 3D world generation ────────────────────────────────────────────────────
from ssv2a.model.generate_3d_scene import (
    generate_object_mesh,
    _boost_vertex_colors,
    create_grid_platform,
    build_3d_world,
    export_scene_glb,
    export_individual_glbs,
    export_scene_json,
)

# ── Optional mm2a ───────────────────────────────────────────────────────────
try:
    from ssv2a.model.pipeline_mm2a import (
        gemini_filter_local_imgs,
        blip_vqa_caption_crops,
        clip_embed_texts_l14,
    )
    MM2A_AVAILABLE = True
except Exception as _e:
    MM2A_AVAILABLE = False
    print(f"[warn] mm2a imports unavailable ({_e}) — only i2a pipeline.")

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_CFG  = "weights/ssv2a.json"
DEFAULT_CKPT = "weights/ssv2a.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_sessions: dict = {}  # session_id → session data

# ═══════════════════════════════════════════════════════════════════════════
# Visual helpers
# ═══════════════════════════════════════════════════════════════════════════

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


def _draw_detections(img, bboxes, labels, depths=None):
    draw = ImageDraw.Draw(img)
    font = _get_font(15)
    for i, (box, label) in enumerate(zip(bboxes, labels)):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=color)
        tag = "[%d] %s" % (i, label)
        if depths:
            tag += " (Z=%.1f)" % depths[i]
        draw.text((x1 + 2, max(y1 - 20, 0)), tag, fill=color, font=font)
    return img


def _draw_camera_with_direction(img, cam_x, cam_y, yaw_deg, pitch_deg):
    """Draw camera icon + arrow on 2D image."""
    draw = ImageDraw.Draw(img)
    r = 22
    draw.line([(cam_x - r, cam_y), (cam_x + r, cam_y)], fill="red", width=3)
    draw.line([(cam_x, cam_y - r), (cam_x, cam_y + r)], fill="red", width=3)
    draw.ellipse([cam_x - r, cam_y - r, cam_x + r, cam_y + r],
                 outline="red", width=3)
    draw.ellipse([cam_x - 5, cam_y - 5, cam_x + 5, cam_y + 5], fill="red")

    arrow_len = 80
    dx = math.sin(math.radians(yaw_deg)) * arrow_len
    dy = -math.sin(math.radians(pitch_deg)) * arrow_len
    if abs(dx) < 1 and abs(dy) < 1:
        dy = -arrow_len * 0.15

    ex, ey = cam_x + dx, cam_y + dy
    draw.line([(cam_x, cam_y), (ex, ey)], fill="red", width=4)
    angle = math.atan2(ey - cam_y, ex - cam_x)
    head = 16
    for offset in [2.5, -2.5]:
        hx = ex - head * math.cos(angle + offset)
        hy = ey - head * math.sin(angle + offset)
        draw.line([(ex, ey), (hx, hy)], fill="red", width=3)

    font = _get_font(13)
    draw.text((cam_x + r + 6, cam_y - 10),
              "CAM  yaw=%+.0f°  pitch=%+.0f°" % (yaw_deg, pitch_deg),
              fill="red", font=font)
    return img


def _make_depth_overlay(image_pil, depth_map, alpha=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    d = depth_map.copy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    colored = (cm.magma(d)[:, :, :3] * 255).astype(np.uint8)
    depth_pil = Image.fromarray(colored).resize(image_pil.size, Image.BILINEAR)
    return Image.blend(image_pil.convert("RGB"), depth_pil, alpha)


def _3d_table_md(objects_3d, spatial_params):
    """Build a markdown table of 3D spatial parameters (HRTF-enhanced)."""
    lines = [
        "| # | Label | Dist | Azim | Elev | Gain | Pan | ITD | ILD | Reverb | Delay |",
        "|---|-------|------|------|------|------|-----|-----|-----|--------|-------|",
    ]
    for i, (obj, sp) in enumerate(zip(objects_3d, spatial_params)):
        lines.append(
            "| %d | %s | %.1f | %+.0f° | %+.0f° | %.2f | %+.2f | %.2fms | %.1fdB | %.0f%% | %.1fms |"
            % (i, obj.label, obj.distance,
               obj.azimuth_deg, obj.elevation_deg,
               sp["gain"], sp["pan"],
               sp.get("itd_s", 0) * 1000,
               sp.get("ild_db", 0),
               sp.get("reverb", 0) * 100,
               sp.get("delay_s", 0) * 1000)
        )
    return "\n".join(lines)


def _locality_table_md(labels, localities, gains, pans):
    """Markdown table for locality-based spatial audio."""
    lines = [
        "| # | Label | Locality | Gain | Gain (dB) | Pan |",
        "|---|-------|----------|------|-----------|-----|",
    ]
    for i in range(len(labels)):
        gain_db = 20 * math.log10(max(gains[i], 1e-8))
        lines.append(
            "| %d | %s | %.3f | %.3f | %+.1f dB | %+.2f |"
            % (i, labels[i], localities[i], gains[i], gain_db, pans[i])
        )
    return "\n".join(lines)


def _locality_stereo_mix(mono_waves, localities, pans, sr=16000):
    """Locality-based stereo mix: bigger bbox = louder.

    Parameters
    ----------
    mono_waves : np.ndarray [N, 1, samples]
    localities : list[float]  — bbox_area / img_area (0–1)
    pans       : list[float]  — -1 (full left) to +1 (full right)
    sr         : int

    Returns
    -------
    stereo : np.ndarray [2, samples]
    """
    n = mono_waves.shape[0]
    samples = mono_waves.shape[-1]

    # Compute gains: normalise so biggest object = 1.0, then power curve
    # to push small/far objects down hard.
    # Exponent 2 means:
    #   biggest object   → gain = 1.0   (full volume)
    #   half-size object → gain = 0.25  (-12 dB, clearly quieter)
    #   tiny object      → gain ≈ 0     (barely audible)
    max_loc = max(localities) if max(localities) > 0 else 1e-6
    norm_locs = [loc / max_loc for loc in localities]
    gains = [nl ** 2.0 for nl in norm_locs]

    stereo = np.zeros((2, samples), dtype=np.float64)
    for i in range(n):
        sig = mono_waves[i, 0].astype(np.float64)
        sig *= gains[i]

        # Constant-power pan
        pan = float(np.clip(pans[i], -1.0, 1.0))
        theta = (pan + 1.0) * math.pi / 4.0  # 0 → π/2
        l_gain = math.cos(theta)
        r_gain = math.sin(theta)

        stereo[0] += sig * l_gain
        stereo[1] += sig * r_gain

    # Normalise to prevent clipping
    peak = max(np.abs(stereo).max(), 1e-8)
    if peak > 0.95:
        stereo *= 0.90 / peak

    return stereo.astype(np.float32)


def _compute_localities_and_pans(bboxes, img_width, img_height):
    """Compute locality (bbox_area / img_area) and pan (center x) per object."""
    img_area = img_width * img_height
    localities = []
    pans = []
    for box in bboxes:
        area = abs(box[2] - box[0]) * abs(box[3] - box[1])
        locality = area / img_area
        localities.append(locality)
        # Pan: center x normalised to [-1, +1]
        cx = (box[0] + box[2]) / 2.0
        pan = (cx / img_width - 0.5) * 2.0
        pans.append(pan)
    return localities, pans


def _project_3d_to_localities(objects_3d, cam_pos, cam_fwd, cam_right, cam_up,
                               hfov_deg, img_width, img_height,
                               ref_sizes=None,
                               original_localities=None):
    """Project 3D objects into a virtual camera view → localities & pans.

    Uses **spread-normalised distance** so that even at large orbit radii
    the closest object gets locality ≈ 1.0 and the farthest drops to ~0.15.
    This ensures the gain ranking flips when you orbit to the other side.

    Parameters
    ----------
    objects_3d : list of Object3D
    cam_pos, cam_fwd, cam_right, cam_up : np.ndarray (3,)
    hfov_deg : float
    img_width, img_height : int
    ref_sizes, original_localities : unused, kept for API compat.

    Returns
    -------
    localities : list[float]
    pans : list[float]
    """
    half_w = math.tan(math.radians(hfov_deg / 2.0))

    raw_dists = []
    raw_pans = []
    behind = []

    for i, obj in enumerate(objects_3d):
        obj_pos = np.array([obj.X, obj.Y, obj.Z])
        to_obj = obj_pos - cam_pos

        # Depth along camera forward axis
        depth = float(np.dot(to_obj, cam_fwd))
        if depth < 0.01:              # behind camera → silent
            raw_dists.append(None)
            raw_pans.append(0.0)
            behind.append(True)
            continue

        behind.append(False)

        # Pan from camera right vector
        x_cam = float(np.dot(to_obj, cam_right))
        pan = float(np.clip(x_cam / (depth * half_w + 1e-12), -1.0, 1.0))
        raw_pans.append(pan)

        # Euclidean distance from camera
        dist = float(np.linalg.norm(to_obj))
        raw_dists.append(max(dist, 1e-6))

    # ── Smooth distance-based locality ──
    # Instead of rank-normalizing (which is binary with 2 objects),
    # measure how much farther each object is relative to the scene size.
    # This gives continuous variation: equidistant → same gain,
    # slightly closer → slightly louder, much closer → much louder.
    valid_dists = [d for d in raw_dists if d is not None]
    if not valid_dists:
        return [0.0] * len(objects_3d), [0.0] * len(objects_3d)

    min_d = min(valid_dists)

    # Scene diameter = max spread between any two objects
    obj_positions = np.array([[o.X, o.Y, o.Z] for o in objects_3d])
    scene_diam = float(np.linalg.norm(obj_positions.max(axis=0) - obj_positions.min(axis=0)))
    scene_diam = max(scene_diam, 1.0)  # avoid div-by-zero for single object

    CONTRAST = 3.0  # higher = more aggressive attenuation with distance

    localities = []
    for i in range(len(objects_3d)):
        if behind[i]:
            localities.append(0.0)
        else:
            # How many "scene diameters" farther is this object vs closest?
            # excess = 0 → same distance as closest → locality ≈ 1.0
            # excess = 1 → one full scene-width farther → locality ≈ 0.25
            excess = (raw_dists[i] - min_d) / scene_diam
            locality = 1.0 / (1.0 + CONTRAST * excess)
            localities.append(float(np.clip(locality, 0.05, 1.0)))

    print("[_project_3d_to_localities] cam_pos=%s" % cam_pos)
    for i, obj in enumerate(objects_3d):
        d = raw_dists[i] if raw_dists[i] is not None else -1
        print("  [%d] %s  dist=%.2f  locality=%.4f  pan=%+.3f  behind=%s"
              % (i, obj.label, d, localities[i], raw_pans[i], behind[i]))

    return localities, raw_pans


# ═══════════════════════════════════════════════════════════════════════════
# Deduplication — remove overlapping detections of the same class
# ═══════════════════════════════════════════════════════════════════════════

def _box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _box_center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _iou(box_a, box_b):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0]); ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2]); yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = _box_area(box_a)
    area_b = _box_area(box_b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _containment(small_box, big_box):
    """Fraction of small_box's area that is inside big_box."""
    xa = max(small_box[0], big_box[0]); ya = max(small_box[1], big_box[1])
    xb = min(small_box[2], big_box[2]); yb = min(small_box[3], big_box[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_s = _box_area(small_box)
    return inter / area_s if area_s > 0 else 0.0


def _center_dist_ratio(box_a, box_b):
    """Distance between box centers as ratio of the average box diagonal."""
    import math
    ca, cb = _box_center(box_a), _box_center(box_b)
    dist = math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
    diag_a = math.sqrt(_box_area(box_a))
    diag_b = math.sqrt(_box_area(box_b))
    avg_diag = (diag_a + diag_b) / 2
    return dist / avg_diag if avg_diag > 0 else float("inf")


def _deduplicate_detections(crops, iou_thresh=0.25):
    """Remove duplicate detections caused by overlapping / hierarchical
    classes (OIv7 may detect the same object as 'Lion' AND 'Carnivore').

    Label-AGNOSTIC: suppresses any two boxes that either:
      - IoU > iou_thresh  (0.25 = fairly aggressive)
      - Smaller box is >= 45% contained in the larger box
      - Centers are within 0.4x average-box-size of each other
    Keeps the larger detection.
    """
    if len(crops) <= 1:
        return crops

    # Sort by box area descending — keep larger detections
    indexed = sorted(enumerate(crops),
                     key=lambda ic: _box_area(ic[1][3]),
                     reverse=True)
    keep = []
    used = set()
    for i, crop_i in indexed:
        if i in used:
            continue
        keep.append(crop_i)
        used.add(i)
        for j, crop_j in indexed:
            if j in used:
                continue
            iou_val = _iou(crop_i[3], crop_j[3])
            cont_val = _containment(crop_j[3], crop_i[3])
            cdist = _center_dist_ratio(crop_i[3], crop_j[3])
            if iou_val > iou_thresh or cont_val > 0.45 or cdist < 0.4:
                print(f"  [dedup] suppressing '{crop_j[2]}' "
                      f"(IoU={iou_val:.2f}, contain={cont_val:.2f}, "
                      f"cdist={cdist:.2f}) — kept '{crop_i[2]}'")
                used.add(j)
    return keep


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Detect + Depth
# ═══════════════════════════════════════════════════════════════════════════

def step_detect_depth(image_pil, cfg_path, ckpt_path, batch_size,
                      depth_model_name, hfov):
    """YOLO detection + depth estimation."""
    if image_pil is None:
        return (
            None, None,
            "⬆ Upload an image first.",
            gr.Slider.update(maximum=512, value=256, interactive=False),
            gr.Slider.update(maximum=512, value=256, interactive=False),
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),
            "",
        )

    cfg_path  = cfg_path  or DEFAULT_CFG
    ckpt_path = ckpt_path or DEFAULT_CKPT
    hfov = float(hfov)

    sid = "s_%d" % int(time.time() * 1000)
    tmp_dir = Path(tempfile.mkdtemp(prefix="3dworld_"))
    img_path = str(tmp_dir / "input.png")
    image_pil.save(img_path, "PNG")
    w, h = image_pil.size

    with open(cfg_path, "r") as fp:
        config_dict = json.load(fp)

    # ── YOLO ──
    local_imgs = detect_spatial(
        [img_path], config_dict["detector"],
        save_dir=tmp_dir / "crops",
        batch_size=int(batch_size), device=DEVICE,
    )
    img_dims = getattr(local_imgs, "img_dimensions", {})
    crops = local_imgs.get(img_path, [])

    # ── Debug: show what YOLO returned ──
    print(f"[YOLO] Raw detections: {len(crops)}")
    for ci, c in enumerate(crops):
        print(f"  [{ci}] label={c[2]!r}  box={[int(v) for v in c[3]]}  "
              f"area={_box_area(c[3]):.0f}")

    # ── Remove duplicate detections (hierarchical classes / overlaps) ──
    before = len(crops)
    crops = _deduplicate_detections(crops, iou_thresh=0.25)
    if len(crops) < before:
        print(f"[dedup] removed {before - len(crops)} duplicate(s) "
              f"({before} → {len(crops)})")
    print(f"[YOLO] After dedup: {len(crops)} — "
          f"{[c[2] for c in crops]}")

    if not crops:
        return (
            np.array(image_pil), None,
            "No objects detected — try a different image.",
            gr.Slider.update(maximum=w, value=w // 2, interactive=False),
            gr.Slider.update(maximum=h, value=h // 2, interactive=False),
            gr.Button.update(interactive=False),
            gr.Button.update(interactive=False),
            sid,
        )

    bboxes = [c[3] for c in crops]
    labels = [c[2] for c in crops]
    crop_paths = [c[0] for c in crops]
    ww, hh = img_dims.get(img_path, (w, h))

    # ── Depth ──
    d_model, d_proc = load_depth_model(depth_model_name, DEVICE)
    depth_map = estimate_depth(img_path, d_model, d_proc, DEVICE,
                               model_name=depth_model_name)
    del d_model, d_proc
    torch.cuda.empty_cache()

    obj_depths = depth_for_bboxes(depth_map, bboxes, method="median")

    # ── Store session ──
    _sessions[sid] = dict(
        img_path=img_path, tmp_dir=tmp_dir,
        config_dict=config_dict,
        image_pil=image_pil.copy(),
        bboxes=bboxes, labels=labels, crop_paths=crop_paths,
        local_imgs=local_imgs,
        img_width=ww, img_height=hh,
        depth_map=depth_map, obj_depths=obj_depths,
        hfov=hfov,
    )

    # ── 2D annotated ──
    annotated = _draw_detections(image_pil.copy(), bboxes, labels, obj_depths)
    depth_overlay = _make_depth_overlay(image_pil, depth_map, alpha=0.45)
    depth_overlay = _draw_detections(depth_overlay, bboxes, labels, obj_depths)

    info_lines = ["### Detected %d object(s)\n" % len(crops)]
    for i, (lbl, box, d) in enumerate(zip(labels, bboxes, obj_depths)):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        info_lines.append(
            "- **[%d] %s** — pixel (%d, %d), depth Z = **%.1f**"
            % (i, lbl, cx, cy, d))
    info_lines.append(
        "\n**Click ② Generate 3D World** to create full 3D meshes, "
        "or adjust Camera then ③ Generate Audio.")

    return (
        np.array(annotated),
        np.array(depth_overlay),
        "\n".join(info_lines),
        gr.Slider.update(maximum=ww, value=ww // 2, interactive=True),
        gr.Slider.update(maximum=hh, value=hh // 2, interactive=True),
        gr.Button.update(interactive=True),  # generate_3d_btn
        gr.Button.update(interactive=True),  # generate_audio_btn (after preview)
        sid,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight viewer GLB — decimated meshes for fast browser loading
# ═══════════════════════════════════════════════════════════════════════════

def _progress_bar(current, total, label="", width=40):
    """Print a terminal progress bar."""
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r  [{bar}] {pct*100:5.1f}%  {label}", end='', flush=True)
    if current >= total:
        print()  # newline at 100%


def _decimate_for_viewer(mesh, target_faces: int = 30_000, label: str = ""):
    """Stride-based face decimation — handles both vertex-color and texture meshes."""
    import numpy as np
    n = len(mesh.faces)
    vis_type = type(mesh.visual).__name__
    print(f"    [DECIMATE] {label}: {n:,} faces, visual={vis_type}, target={target_faces:,}")

    if n <= target_faces:
        print(f"    [DECIMATE] {label}: already under target — copying as-is")
        return mesh.copy()

    stride = max(2, int(np.ceil(n / target_faces)))
    kept_faces = mesh.faces[::stride]
    print(f"    [DECIMATE] {label}: stride={stride}, kept {len(kept_faces):,} faces")

    # Detect visual type
    has_vc = (hasattr(mesh.visual, 'vertex_colors')
              and mesh.visual.vertex_colors is not None
              and isinstance(mesh.visual, trimesh.visual.ColorVisuals))

    if has_vc:
        print(f"    [DECIMATE] {label}: vertex-color mode")
        light = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=kept_faces,
            vertex_colors=mesh.visual.vertex_colors.copy(),
            process=True,
        )
    else:
        print(f"    [DECIMATE] {label}: texture mode → flat colour + trying to_color()")
        light = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=kept_faces,
            process=True,
        )
        try:
            if hasattr(mesh.visual, 'to_color'):
                vc = mesh.visual.to_color()
                if vc.vertex_colors is not None:
                    unique_verts = np.unique(kept_faces.ravel())
                    full_vc = vc.vertex_colors
                    light_vc = np.full((len(mesh.vertices), 4), 180, dtype=np.uint8)
                    light_vc[:len(full_vc)] = full_vc
                    light = trimesh.Trimesh(
                        vertices=mesh.vertices.copy(),
                        faces=kept_faces,
                        vertex_colors=light_vc,
                        process=True,
                    )
                    print(f"    [DECIMATE] {label}: baked vertex colours from texture ✓")
        except Exception as e:
            print(f"    [DECIMATE] {label}: to_color() failed ({e}), using flat grey")

    light.remove_unreferenced_vertices()
    print(f"    [DECIMATE] {label}: final {len(light.faces):,} faces, {len(light.vertices):,} verts ✓")
    return light


def _build_viewer_scene(meshes, objects_3d, camera_fx, camera_fy,
                        target_faces_per_mesh: int = 30_000):
    """Build a decimated scene trimesh for the model-viewer."""
    from ssv2a.model.generate_3d_scene import _estimate_mesh_scale
    n = len(meshes)
    print(f"\n{'='*60}")
    print(f"  [VIEWER] Building decimated viewer scene ({n} objects)")
    print(f"{'='*60}")
    scene = trimesh.Scene()

    for i, (obj, mesh) in enumerate(zip(objects_3d, meshes)):
        _progress_bar(i, n, f"Decimating [{i}] {obj.label}")
        try:
            light = _decimate_for_viewer(mesh, target_faces_per_mesh,
                                         label=f"[{i}] {obj.label}")
        except Exception as e:
            print(f"\n  [VIEWER] ✗ Decimation FAILED for [{i}] {obj.label}: {e}")
            print(f"  [VIEWER]   → using full mesh as fallback")
            light = mesh
        scale = _estimate_mesh_scale(obj.bbox, obj.depth_value,
                                     camera_fx, camera_fy)
        scale = max(0.5, min(scale, 50.0))
        # Match build_3d_world: center → scale → translate
        light.vertices -= light.vertices.mean(axis=0)
        light.vertices *= scale
        light.vertices += np.array([obj.X, obj.Y, obj.Z])
        scene.add_geometry(light,
                           node_name=f"{i:02d}_{obj.label}")
    _progress_bar(n, n, "All objects decimated")

    total_faces = sum(len(g.faces) for g in scene.geometry.values())
    total_verts = sum(len(g.vertices) for g in scene.geometry.values())
    print(f"  [VIEWER] Scene totals: {total_verts:,} verts, {total_faces:,} faces")
    return scene


# ═══════════════════════════════════════════════════════════════════════════
# Unity-style Three.js viewer — inline HTML generator
# ═══════════════════════════════════════════════════════════════════════════

def _build_unity_viewer_html(scene_glb_path, objects_3d, meshes,
                              camera_fx, camera_fy,
                              img_width=512, img_height=512):
    """Build self-contained HTML with <model-viewer> for 3D navigation.

    Uses Google's <model-viewer> web component — loads GLBs natively,
    orbit controls built-in, very fast.  Camera orbit (theta/phi/radius)
    is mapped to the audio pipeline's yaw/pitch + 2D pixel coords.

    Because Gradio's gr.HTML strips <script> tags (DOMPurify), we write
    the full page to a .html file and return an <iframe> pointing to it.
    The iframe is same-origin so parent.document access works for
    pushing camera values into the hidden Gradio Number fields.
    """

    # Build object metadata JSON
    obj_data = []
    for i, obj in enumerate(objects_3d):
        obj_data.append({
            "index": i, "label": obj.label,
            "X": round(float(obj.X), 2),
            "Y": round(float(obj.Y), 2),
            "Z": round(float(obj.Z), 2),
            "distance": round(float(obj.distance), 2),
            "azimuth_deg": round(float(obj.azimuth_deg), 1),
            "elevation_deg": round(float(obj.elevation_deg), 1),
        })

    objects_json = json.dumps(obj_data)

    # GLB URL via our custom /viewer_asset route (Gradio blocks .glb from /tmp/3dworld_*)
    import urllib.parse as _up2
    glb_url = f"/viewer_asset?path={_up2.quote(scene_glb_path, safe='')}"

    # ── Full standalone HTML page ────────────────────────────────────────
    full_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>3D World Viewer</title>
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#1a1a2e; overflow:hidden; font-family:system-ui,sans-serif; }}
</style>
</head>
<body>
<div id="mv-wrapper" style="position:relative;width:100%;height:100vh;overflow:hidden;">

  <model-viewer id="mv-viewer"
    src="{glb_url}"
    alt="3D World"
    camera-controls
    touch-action="pan-y"
    shadow-intensity="0.8"
    exposure="1.1"
    style="width:100%;height:100%;background:#1a1a2e;--poster-color:#1a1a2e;"
    loading="eager"
    camera-orbit="0deg 60deg auto"
    min-camera-orbit="auto auto auto"
    max-camera-orbit="auto auto auto"
    interaction-prompt="auto"
  >
    <div slot="progress-bar" id="mv-progress"
         style="position:absolute;bottom:0;left:0;height:4px;background:linear-gradient(90deg,#4a6cf7,#6366f1);
                transition:width 0.3s;width:0%;"></div>
  </model-viewer>

  <!-- HUD overlay -->
  <div style="position:absolute;top:10px;left:10px;z-index:10;pointer-events:none;">
    <div style="background:rgba(10,10,30,0.88);border:1px solid rgba(100,140,255,0.3);border-radius:8px;
                padding:10px 14px;backdrop-filter:blur(6px);font-size:13px;line-height:1.7;
                color:#e0e0e0;pointer-events:auto;">
      <div style="color:#7da8ff;font-weight:700;font-size:14px;margin-bottom:4px;">📷 Camera</div>
      <div>Orbit: <span id="mv-orbit" style="font-family:Consolas,monospace;color:#90ee90;">0°, 60°</span></div>
      <div>Radius: <span id="mv-radius" style="font-family:Consolas,monospace;color:#90ee90;">auto</span></div>
      <div>→ Yaw: <span id="mv-yaw" style="font-family:Consolas,monospace;color:#ffd700;">0°</span>
           Pitch: <span id="mv-pitch" style="font-family:Consolas,monospace;color:#ffd700;">0°</span></div>
      <div style="margin-top:6px;color:#7da8ff;font-weight:700;">🎯 Objects ({len(obj_data)})</div>
      <div id="mv-objs" style="font-size:12px;"></div>
    </div>
  </div>

  <!-- Controls help -->
  <div style="position:absolute;bottom:56px;left:50%;transform:translateX(-50%);z-index:10;pointer-events:none;
              background:rgba(10,10,30,0.85);border:1px solid rgba(100,140,255,0.3);border-radius:8px;
              padding:6px 16px;font-size:11px;color:#aaa;backdrop-filter:blur(6px);text-align:center;white-space:nowrap;">
    <b>Drag</b> to rotate &nbsp;|&nbsp; <b>Scroll</b> to zoom &nbsp;|&nbsp;
    <b>RMB / two-finger</b> to pan
  </div>

  <!-- Send camera button -->
  <button id="mv-send"
    style="position:absolute;bottom:10px;right:10px;z-index:10;
           background:linear-gradient(135deg,#4a6cf7,#6366f1);border:none;border-radius:8px;
           padding:10px 18px;color:#fff;cursor:pointer;font-size:13px;font-weight:600;
           box-shadow:0 2px 8px rgba(99,102,241,0.4);">
    📍 Use This Camera
  </button>
</div>

<script>
(function() {{
  const OBJECTS = {objects_json};
  const FX = {camera_fx:.2f}, FY = {camera_fy:.2f};
  const IMG_W = {img_width}, IMG_H = {img_height};

  function init() {{
    const mv = document.getElementById('mv-viewer');
    if (!mv) return;

    // Object tags
    const objsEl = document.getElementById('mv-objs');
    objsEl.innerHTML = OBJECTS.map(o =>
      '<span style="display:inline-block;background:rgba(100,140,255,0.2);border:1px solid rgba(100,140,255,0.4);' +
      'border-radius:4px;padding:1px 6px;margin:1px;font-size:11px;">[' + o.index + '] ' + o.label + '</span>'
    ).join('');

    // Progress bar
    mv.addEventListener('progress', (e) => {{
      const bar = document.getElementById('mv-progress');
      if (bar) bar.style.width = (e.detail.totalProgress * 100) + '%';
    }});
    mv.addEventListener('load', () => {{
      const bar = document.getElementById('mv-progress');
      if (bar) bar.style.display = 'none';
    }});

    // HUD update on camera change
    mv.addEventListener('camera-change', () => {{
      const orbit = mv.getCameraOrbit();
      const thetaDeg = (orbit.theta * 180 / Math.PI);
      const phiDeg   = (orbit.phi * 180 / Math.PI);
      const yaw   = Math.round(thetaDeg);
      const pitch  = Math.round(90 - phiDeg);

      document.getElementById('mv-orbit').textContent =
        thetaDeg.toFixed(1) + '°, ' + phiDeg.toFixed(1) + '°';
      document.getElementById('mv-radius').textContent =
        (typeof orbit.radius === 'number' ? orbit.radius.toFixed(1) : 'auto');
      document.getElementById('mv-yaw').textContent = yaw + '°';
      document.getElementById('mv-pitch').textContent = pitch + '°';
    }});

    // Helper: set a Gradio Number component value by elem_id
    // We reach INTO the parent frame (Gradio page) via window.parent
    function setGradioInput(elemId, value) {{
      try {{
        const parentDoc = window.parent.document;
        const wrapper = parentDoc.getElementById(elemId);
        if (!wrapper) {{ console.warn('Gradio field not found:', elemId); return; }}
        const inp = wrapper.querySelector('input');
        if (!inp) return;
        const nativeSetter = Object.getOwnPropertyDescriptor(
          window.parent.HTMLInputElement.prototype, 'value'
        ).set;
        nativeSetter.call(inp, value);
        inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
        inp.dispatchEvent(new Event('change', {{ bubbles: true }}));
      }} catch(e) {{
        console.error('Cannot set Gradio field', elemId, e);
      }}
    }}

    // Send camera button
    document.getElementById('mv-send').addEventListener('click', () => {{
      const orbit = mv.getCameraOrbit();
      const thetaDeg = (orbit.theta * 180 / Math.PI);
      const phiDeg   = (orbit.phi * 180 / Math.PI);
      const radius   = orbit.radius;

      let yaw   = Math.round(thetaDeg);
      let pitch  = Math.round(90 - phiDeg);
      yaw   = Math.max(-180, Math.min(180, yaw));
      pitch = Math.max(-60, Math.min(60, pitch));

      // Push raw orbit params into hidden Gradio fields.
      // Python will convert orbit → 3D listener position → per-object distances.
      setGradioInput('unity_cam_x', thetaDeg.toFixed(2));   // orbit theta (deg)
      setGradioInput('unity_cam_y', phiDeg.toFixed(2));     // orbit phi (deg)
      setGradioInput('unity_cam_z', radius ? radius.toFixed(2) : '20');  // orbit radius
      setGradioInput('unity_cam_yaw', yaw);
      setGradioInput('unity_cam_pitch', pitch);

      // Auto-click the Gradio "Apply 3D Camera" button so
      // step_apply_unity_camera runs and sess["listener_pos"] gets set.
      setTimeout(() => {{
        try {{
          const parentDoc = window.parent.document;
          // Find the Gradio button by its text content
          const buttons = parentDoc.querySelectorAll('button');
          for (const b of buttons) {{
            if (b.textContent.includes('Apply 3D Camera')) {{
              b.click();
              console.log('Auto-clicked Apply 3D Camera button');
              break;
            }}
          }}
        }} catch(e) {{
          console.error('Could not auto-click Apply 3D Camera:', e);
        }}
      }}, 300);  // small delay to let Gradio register the input changes

      // Visual feedback
      const btn = document.getElementById('mv-send');
      btn.textContent = '✅ Set! θ=' + thetaDeg.toFixed(0) + '° φ=' + phiDeg.toFixed(0) + '° r=' + (radius||20).toFixed(1);
      btn.style.background = 'linear-gradient(135deg,#22c55e,#16a34a)';
      setTimeout(() => {{
        btn.textContent = '📍 Use This Camera';
        btn.style.background = 'linear-gradient(135deg,#4a6cf7,#6366f1)';
      }}, 2000);
    }});
  }}

  // model-viewer may load async, retry
  if (customElements.get('model-viewer')) {{
    init();
  }} else {{
    let tries = 0;
    const poll = setInterval(() => {{
      tries++;
      if (customElements.get('model-viewer') || tries > 50) {{
        clearInterval(poll);
        init();
      }}
    }}, 200);
  }}
}})();
</script>
</body>
</html>'''

    # ── Write to file next to the GLB ────────────────────────────────────
    viewer_dir = os.path.dirname(scene_glb_path)
    viewer_html_path = os.path.join(viewer_dir, "viewer.html")
    with open(viewer_html_path, "w") as f:
        f.write(full_html)
    print(f"  [VIEWER] Wrote {viewer_html_path} ({len(full_html):,} chars)")

    # ── Return an <iframe> served by our custom /viewer_page route ────────
    # Gradio's /file= endpoint blocks .html for security, so we serve via
    # a dedicated FastAPI route instead.
    import urllib.parse as _up
    iframe_url = f"/viewer_page?path={_up.quote(viewer_html_path, safe='')}"
    return (
        f'<iframe src="{iframe_url}" '
        f'style="width:100%;height:680px;border:none;border-radius:12px;" '
        f'allow="autoplay;fullscreen" loading="eager"></iframe>'
    )


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Generate 3D World (BLIP + SD Turbo + Hunyuan3D-2)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def step_generate_3d_world(sid, mesh_resolution, hfov):
    """SAM segment → Hunyuan3D-2 mesh → 3D world."""
    if not sid or sid not in _sessions:
        return None, None, "Run detection first."

    sess = _sessions[sid]
    hfov = float(hfov)
    resolution = int(mesh_resolution)  # passed to Hunyuan3D diffusion steps

    image_pil = sess["image_pil"]
    bboxes = sess["bboxes"]
    labels = sess["labels"]
    obj_depths = sess["obj_depths"]
    ww, hh = sess["img_width"], sess["img_height"]

    cam = CameraIntrinsics.from_image_size(ww, hh, hfov_deg=hfov)
    objects_3d = backproject_objects(
        bboxes, labels, obj_depths,
        ww, hh, camera=cam, hfov_deg=hfov,
    )

    # Generate 3D meshes: SAM crop → Hunyuan3D-2 (direct)
    meshes = []
    log_lines = ["### Generative 3D World (YOLO + SAM + Hunyuan3D-2)\n"]
    total_t = time.time()

    # ── Phase 1: SAM-segment every object from the full image ──
    n_obj = len(objects_3d)
    print(f"\n{'='*60}")
    print(f"  STEP ② — GENERATE 3D WORLD  ({n_obj} objects)")
    print(f"{'='*60}")
    print(f"\n  Phase 1/5: SAM Segmentation")
    from ssv2a.model.generate_3d_scene import segment_object, unload_sam
    isolated_crops = []
    for i, (obj, bbox) in enumerate(zip(objects_3d, bboxes)):
        _progress_bar(i, n_obj, f"SAM [{i}] {obj.label}")
        isolated = segment_object(image_pil, bbox, device=DEVICE)
        print(f"    [{i}] SAM-isolated '{obj.label}' "
              f"({isolated.size[0]}x{isolated.size[1]})")
        isolated_crops.append(isolated)
    _progress_bar(n_obj, n_obj, "SAM done")
    # Free SAM VRAM before heavy 3D generation
    unload_sam()
    torch.cuda.empty_cache()

    # ── Phase 2: Generate 3D meshes one by one ──
    print(f"\n  Phase 2/5: Hunyuan3D-2 Mesh Generation")
    for i, (obj, bbox, crop) in enumerate(
            zip(objects_3d, bboxes, isolated_crops)):
        _progress_bar(i, n_obj, f"Generating [{i}] {obj.label}")
        print(f"    [{i}] Starting Hunyuan3D-2 for '{obj.label}' ...")
        t0 = time.time()
        mesh = generate_object_mesh(
            obj.label, image_crop=crop,
            device=DEVICE, resolution=resolution,
        )
        dt = time.time() - t0
        meshes.append(mesh)
        print(f"    [{i}] ✓ {obj.label}: {len(mesh.vertices):,} verts, "
              f"{len(mesh.faces):,} faces ({dt:.1f}s)")
        log_lines.append(
            "- **[%d] %s** → %d vertices, %d faces (%.1fs)"
            % (i, obj.label, len(mesh.vertices), len(mesh.faces), dt)
        )
    _progress_bar(n_obj, n_obj, "All meshes generated")
    torch.cuda.empty_cache()

    # Grid platform (Tinkercad style)
    print(f"\n  Phase 3/5: Assembling 3D World")
    print(f"    Creating grid platform ...")
    max_z = max(o.Z for o in objects_3d) if objects_3d else 50
    max_x = max(abs(o.X) for o in objects_3d) if objects_3d else 30
    platform = create_grid_platform(
        extent_x=max(max_x * 2, 60),
        extent_z=max(max_z * 1.5, 80),
    )
    log_lines.append(
        "- **Platform** → %d vertices" % len(platform.vertices)
    )
    print(f"    Platform: {len(platform.vertices):,} verts")

    print(f"    Building full scene (position + scale meshes) ...")
    scene = build_3d_world(
        meshes, objects_3d,
        camera_fx=cam.fx, camera_fy=cam.fy,
        platform=platform,
        boost_colors=True,
    )

    # Export
    out_dir = str(sess["tmp_dir"])
    scene_path = os.path.join(out_dir, "3d_world.glb")
    print(f"    Exporting full scene GLB → {scene_path}")
    export_scene_glb(scene, scene_path)
    full_mb = os.path.getsize(scene_path) / (1024 * 1024)
    print(f"    ✓ Full scene GLB: {full_mb:.1f} MB")

    for m in meshes:
        _boost_vertex_colors(m, brightness=1.4, saturation=1.15)
    indiv_paths = export_individual_glbs(
        meshes, labels,
        os.path.join(out_dir, "meshes"),
    )

    total_dt = time.time() - total_t
    log_lines.append(
        "\n**Total: %d objects + platform in %.1fs** | Scene: %.0f KB"
        % (len(meshes), total_dt,
           os.path.getsize(scene_path) / 1024)
    )
    log_lines.append(
        "\n**Rotate / zoom / pan** the 3D world to explore.\n"
        "Each object is a **generated** 3D mesh "
        "(SAM segmentation → Hunyuan3D-2 shape + PBR texture).\n"
        "Grid platform for spatial orientation."
    )

    sess["meshes"] = meshes
    sess["objects_3d"] = objects_3d
    sess["scene_path"] = scene_path
    sess["indiv_mesh_paths"] = indiv_paths

    # ── Export scene.json for unity viewer ──
    print(f"\n  Phase 4/5: Export Metadata")
    cam = CameraIntrinsics.from_image_size(
        sess["img_width"], sess["img_height"], hfov
    )
    scene_json_path = os.path.join(out_dir, "scene.json")
    print(f"    Exporting scene.json ...")
    export_scene_json(
        objects_3d, meshes,
        scene_path, indiv_paths,
        scene_json_path,
        camera_fx=cam.fx, camera_fy=cam.fy,
    )
    sess["scene_json_path"] = scene_json_path
    sess["scene_dir"] = out_dir
    print(f"    ✓ scene.json written")
    log_lines.append("\n**Unity Viewer** data exported (scene.json + meshes).")

    first_mesh_path = indiv_paths[0] if indiv_paths else None

    # ── Build lightweight viewer GLB (decimated for fast browser loading) ──
    print(f"\n  Phase 5/5: Decimated Viewer GLB")
    viewer_glb_path = os.path.join(out_dir, "viewer.glb")
    try:
        t_dec = time.time()
        viewer_scene = _build_viewer_scene(meshes, objects_3d, cam.fx, cam.fy)
        print(f"  [VIEWER] Exporting viewer.glb ...")
        export_scene_glb(viewer_scene, viewer_glb_path)
        viewer_kb = os.path.getsize(viewer_glb_path) // 1024
        dt_dec = time.time() - t_dec
        print(f"  [VIEWER] ✓ viewer.glb: {viewer_kb:,} KB ({dt_dec:.1f}s)")
        log_lines.append(f"**Viewer GLB** (decimated): {viewer_kb} KB")
    except Exception as _ve:
        import traceback
        print(f"\n  [VIEWER] ✗ Decimation FAILED: {_ve}")
        traceback.print_exc()
        print(f"  [VIEWER] → falling back to full GLB ({full_mb:.1f} MB)")
        viewer_glb_path = scene_path

    # Build inline model-viewer HTML
    print(f"\n  Building model-viewer HTML ...")
    unity_html = _build_unity_viewer_html(viewer_glb_path, objects_3d, meshes,
                                           cam.fx, cam.fy,
                                           img_width=ww, img_height=hh)
    print(f"  ✓ HTML ready ({len(unity_html):,} chars)")
    print(f"  ✓ Viewer GLB URL: /file={viewer_glb_path}")
    print(f"\n{'='*60}")
    print(f"  STEP ② COMPLETE — returning to Gradio")
    print(f"{'='*60}\n")

    return scene_path, first_mesh_path, "\n".join(log_lines), unity_html


# ═══════════════════════════════════════════════════════════════════════════
# Step 2½ — Apply Unity Camera → Audio Sliders
# ═══════════════════════════════════════════════════════════════════════════

def step_apply_unity_camera(sid, ucam_x, ucam_y, ucam_z,
                            ucam_yaw, ucam_pitch, hfov):
    """Convert 3D viewer orbit into spatial audio params by working
    entirely in GLB coordinates (the same space model-viewer uses).

    Instead of trying to convert between coordinate systems, we:
    1. Compute the camera's Cartesian position in GLB space using
       model-viewer's own orbit formula.
    2. Compute Euclidean distance from camera to each object.
    3. Normalise the distances and apply strong contrast so that even
       small real distance differences (orbit cameras are far away)
       produce large gain differences.
    4. Compute pan from the camera's right-vector projection.
    """
    if not sid or sid not in _sessions:
        return 256, 256, 0, 0, "Run detection + generate 3D world first."

    sess = _sessions[sid]
    ww, hh = sess["img_width"], sess["img_height"]

    theta_deg = float(ucam_x)   # orbit azimuth (degrees)
    phi_deg   = float(ucam_y)   # orbit polar   (degrees)
    radius    = float(ucam_z)   # orbit radius  (GLB units)
    yaw   = max(-180, min(180, int(round(float(ucam_yaw)))))
    pitch = max(-60,  min(60,  int(round(float(ucam_pitch)))))

    objects_3d = sess.get("objects_3d", [])
    if not objects_3d:
        return ww // 2, hh // 2, 0, 0, "No objects – generate 3D world first."

    # ── 1. Camera position in GLB coordinates ──
    # model-viewer orbit formula (right-handed, Y-up):
    theta_rad = math.radians(theta_deg)
    phi_rad   = math.radians(phi_deg)

    positions = np.array([[o.X, o.Y, o.Z] for o in objects_3d])
    centroid  = positions.mean(axis=0)

    # model-viewer orbits around the scene bounding-box centre ≈ centroid
    cam = np.array([
        centroid[0] + radius * math.sin(phi_rad) * math.sin(theta_rad),
        centroid[1] + radius * math.cos(phi_rad),
        centroid[2] + radius * math.sin(phi_rad) * math.cos(theta_rad),
    ])

    # ── 2. Forward & right vectors (camera looks at centroid) ──
    fwd = centroid - cam
    fwd_len = np.linalg.norm(fwd)
    fwd = fwd / (fwd_len + 1e-12)

    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, world_up)
    r_len = np.linalg.norm(right)
    if r_len < 1e-6:                       # camera directly above/below
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / r_len

    # ── 3. Per-object distance, gain (physics-based), and pan ──
    raw_dists = np.array([
        float(np.linalg.norm(np.array([o.X, o.Y, o.Z]) - cam))
        for o in objects_3d
    ])

    # Reference distance = scene diameter (max spread of objects).
    # Objects at ref_d get gain_dist ≈ 0.5; closer → higher, farther → lower.
    scene_spread = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    ref_d = max(scene_spread, 1e-3)   # avoid zero for single-object scenes

    spatial_params = []
    objs_relative  = []     # lightweight copies for table display

    for i, obj in enumerate(objects_3d):
        d = raw_dists[i]

        # ── Distance attenuation (inverse-distance, physics-motivated) ──
        # gain_dist = ref_d / (ref_d + d)
        #   d ≈ 0        → gain_dist ≈ 1.0
        #   d = ref_d     → gain_dist = 0.5
        #   d = 3·ref_d   → gain_dist = 0.25
        # Camera in the middle → all objects similar distance → similar gain.
        gain_dist = float(ref_d / (ref_d + d + 1e-12))

        # ── Directional factor (front / back of camera) ──
        to_obj = np.array([obj.X, obj.Y, obj.Z]) - cam
        to_obj_n = to_obj / (np.linalg.norm(to_obj) + 1e-12)
        cos_angle = float(np.dot(to_obj_n, fwd))  # -1 = behind, +1 = in front
        # Smooth: objects directly in front ×1.0, to the side ×0.7, behind ×0.4
        gain_dir = float(0.7 + 0.3 * cos_angle)

        # ── Combined gain ──
        gain = float(np.clip(gain_dist * gain_dir, 0.05, 1.0))

        # Pan: signed projection onto camera right vector
        pan = float(np.clip(np.dot(to_obj_n, right), -1.0, 1.0))

        # Reverb: scale with distance relative to scene
        reverb = float(np.clip(0.10 + 0.50 * (d / (ref_d + 1e-12)), 0.10, 0.70))

        # Approximate azimuth for HRTF rendering
        az_deg = float(math.degrees(math.asin(np.clip(pan, -1, 1))))

        sp = {
            "label":          obj.label,
            "gain":           gain,
            "pan":            pan,
            "reverb":         reverb,
            "distance":       float(d),
            "X": float(obj.X), "Y": float(obj.Y), "Z": float(obj.Z),
            "azimuth_deg":    az_deg,
            "elevation_deg":  0.0,
            "elevation_lp_hz": 6000.0,
            "ild_db":         float(abs(pan) * 8.0),
            "delay_s":        float(d / 343.0),
            "itd_s":          float(pan * 0.0007),
        }
        spatial_params.append(sp)

        # Create an Object3D-like namedtuple for _3d_table_md
        from copy import copy as _cp
        orel = _cp(obj)
        orel.distance     = float(d)
        orel.azimuth_deg  = az_deg
        orel.elevation_deg = 0.0
        objs_relative.append(orel)

    sess["objects_3d_rotated"] = objs_relative
    sess["spatial_params"]     = spatial_params
    sess["listener_pos"]       = tuple(cam.tolist())

    # Map to 2D slider coords for the preview overlay
    sx = max(0, min(ww, int(round(ww / 2 + (theta_deg / 90) * (ww / 2)))))
    sy = max(0, min(hh, int(round(hh / 2 - (pitch / 60) * (hh / 2)))))

    # ── Info table ──
    cam_s = f"({cam[0]:.1f}, {cam[1]:.1f}, {cam[2]:.1f})"
    cen_s = f"({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})"
    lines = ["### 📍 3D Camera Applied\n"]
    lines.append(f"**Camera**: {cam_s}  **Scene centre**: {cen_s}")
    lines.append(f"**Orbit**: θ={theta_deg:.0f}°  φ={phi_deg:.0f}°  r={radius:.1f}")
    lines.append(f"**Yaw**: {yaw}°  **Pitch**: {pitch}°\n")

    lines.append("| Object | Position | Distance | Gain | Pan | Reverb |")
    lines.append("|--------|----------|----------|------|-----|--------|")
    for sp in spatial_params:
        pos = f"({sp['X']:.1f}, {sp['Y']:.1f}, {sp['Z']:.1f})"
        lines.append(
            f"| **{sp['label']}** | {pos} | {sp['distance']:.1f} | "
            f"{sp['gain']:.2f} | {sp['pan']:+.2f} | {sp['reverb']*100:.0f}% |"
        )
    lines.append("\n_Gain = inverse-distance × directional (front 1.0 → behind 0.4)._")

    info = "\n".join(lines)
    return sx, sy, yaw, pitch, info


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Preview listener (spatial params)
# ═══════════════════════════════════════════════════════════════════════════

def step_preview_listener(sid, cam_x, cam_y, yaw, pitch,
                          hfov, attenuation, enable_delay):
    """Compute 3D spatial audio params for current camera settings.

    If the user has already applied a 3D camera position (from the Unity
    viewer), use the stored listener-relative positions.  Otherwise,
    fall back to the 2D slider-based backprojection.
    """
    if not sid or sid not in _sessions:
        return None, "Run detection first."

    sess = _sessions[sid]
    cam_x, cam_y = float(cam_x), float(cam_y)
    yaw, pitch = float(yaw), float(pitch)
    hfov = float(hfov)
    attenuation = float(attenuation)

    ww, hh = sess["img_width"], sess["img_height"]

    # If we already have listener-relative spatial params from the 3D viewer,
    # use those — they encode the correct per-object distances.
    if "listener_pos" in sess and "spatial_params" in sess:
        objects_3d_rotated = sess["objects_3d_rotated"]
        spatial_params = sess["spatial_params"]
        # Still draw 2D overlay
        img = sess["image_pil"].copy()
        img = _draw_detections(img, sess["bboxes"], sess["labels"], sess["obj_depths"])
        img = _draw_camera_with_direction(img, cam_x, cam_y, yaw, pitch)
        lp = sess["listener_pos"]
        table = _3d_table_md(objects_3d_rotated, spatial_params)
        md = (
            "### 3D Listener at (%.1f, %.1f, %.1f)  |  Yaw %+.0f°  |  Pitch %+.0f°\n\n%s"
            % (lp[0], lp[1], lp[2], yaw, pitch, table)
        )
        return np.array(img), md

    # Fallback: 2D slider-based backprojection
    camera = CameraIntrinsics.from_image_size(ww, hh, hfov_deg=hfov)
    camera.cx = cam_x
    camera.cy = cam_y

    objects_3d_listener = backproject_objects(
        sess["bboxes"], sess["labels"], sess["obj_depths"],
        ww, hh, camera=camera, hfov_deg=hfov,
    )
    objects_3d_rotated = apply_listener_rotation(objects_3d_listener, yaw, pitch)
    spatial_params = compute_3d_spatial_params(
        objects_3d_rotated,
        attenuation_exp=attenuation,
        enable_delay=enable_delay,
    )

    sess["camera"] = camera
    sess["objects_3d_rotated"] = objects_3d_rotated
    sess["spatial_params"] = spatial_params

    img = sess["image_pil"].copy()
    img = _draw_detections(img, sess["bboxes"], sess["labels"], sess["obj_depths"])
    img = _draw_camera_with_direction(img, cam_x, cam_y, yaw, pitch)

    table = _3d_table_md(objects_3d_rotated, spatial_params)
    md = (
        "### Camera at (%d, %d)  |  Yaw %+.0f°  |  Pitch %+.0f°\n\n%s"
        % (cam_x, cam_y, yaw, pitch, table)
    )

    return np.array(img), md


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Generate Audio
# ═══════════════════════════════════════════════════════════════════════════

def step_generate_audio(sid, pipeline, cfg_path, ckpt_path,
                        batch_size, var_samples, cycle_its, cycle_samples,
                        duration, seed,
                        cam_x, cam_y, yaw, pitch,
                        hfov, attenuation, enable_delay):
    """Full-quality per-object audio: each crop runs the complete cycle_mix.

    Pipeline:
      1. Reference: image_to_audio() on the full image (exact infer_i2a).
      2. Per-object: image_to_audio() on EACH crop individually.
         Each crop goes through the full cycle_mix (64 its × 64 samples)
         producing maximum-quality CLAP → AudioLDM mono waveform.
      3. Locality-weighted stereo mix:
         - gain = sqrt(locality)   (bigger bbox = closer = louder)
         - pan  = bbox center x    (left in image → left speaker)
      4. Tracks cached for instant re-rendering when camera moves.
    """
    import soundfile as _sf

    if not sid or sid not in _sessions:
        return None, None, "Run detection first."

    sess = _sessions[sid]

    set_seed(int(seed))
    cfg_path  = cfg_path  or DEFAULT_CFG
    ckpt_path = ckpt_path or DEFAULT_CKPT
    config_dict  = sess["config_dict"]
    img_path     = sess["img_path"]
    labels       = sess["labels"]
    crop_paths   = sess["crop_paths"]
    bboxes       = sess["bboxes"]
    ww, hh       = sess["img_width"], sess["img_height"]

    bs  = int(batch_size)
    vs  = int(var_samples)
    ci  = int(cycle_its)
    cs  = int(cycle_samples)
    dur = int(duration)

    use_mm2a = pipeline == "mm2a (LLM filter + BLIP)"
    log = []

    active_crops  = list(crop_paths)
    active_labels = list(labels)
    active_bboxes = list(bboxes)

    # ── optional mm2a filter ──
    if use_mm2a:
        if not MM2A_AVAILABLE:
            return None, None, "mm2a imports failed — only i2a available."
        log.append("LLM sound-source filter …")
        local_3t = {img_path: [(cp, 0.0, lbl)
                               for cp, lbl in zip(crop_paths, labels)]}
        filtered, _ = gemini_filter_local_imgs(local_3t)
        kept = filtered.get(img_path, [])
        if not kept:
            return None, None, "LLM filter dropped ALL objects."
        kept_set = {cp for cp, _, _ in kept}
        active_crops, active_labels, active_bboxes = [], [], []
        for i, cp in enumerate(crop_paths):
            if cp in kept_set:
                active_crops.append(cp)
                active_labels.append(labels[i])
                active_bboxes.append(bboxes[i])
        log.append("Kept %d / %d objects" % (len(active_crops), len(crop_paths)))

    n = len(active_crops)

    # ────────────────────────────────────────────────────────────────
    # 1. Reference: full image → image_to_audio (exact infer_i2a)
    # ────────────────────────────────────────────────────────────────
    log.append("[1/2] Reference audio (full image, full cycle_mix) …")
    ref_save_dir = sess["tmp_dir"] / "ref_audio"
    i2a_image_to_audio(
        [img_path], text="", transcription="",
        save_dir=str(ref_save_dir), config=cfg_path,
        gen_remix=True, gen_tracks=False, emb_only=False,
        pretrained=ckpt_path, batch_size=bs, var_samples=vs,
        shuffle_remix=True, cycle_its=ci, cycle_samples=cs,
        keep_data_cache=False, duration=dur, seed=int(seed),
        device=DEVICE,
    )
    torch.cuda.empty_cache()

    ref_wav_path = str(ref_save_dir / (Path(img_path).stem + ".wav"))
    ref_wave, ref_sr = _sf.read(ref_wav_path)
    if ref_wave.ndim > 1:
        ref_wave = ref_wave[:, 0]
    ref_wave = np.array(ref_wave, dtype=np.float32)
    log.append("  Reference done.  Peak=%.4f" % float(np.abs(ref_wave).max()))

    # ────────────────────────────────────────────────────────────────
    # 2. Per-object: run full image_to_audio on each crop
    #    Pass the crop as a pre-computed dict so YOLO is skipped and
    #    the crop itself is its own "detection" → full cycle_mix path.
    # ────────────────────────────────────────────────────────────────
    log.append("[2/2] Per-object full cycle_mix (%d objects) …" % n)
    tracks_dir = sess["tmp_dir"] / "tracks"
    tracks_dir.mkdir(exist_ok=True)

    per_object_waves_list = []
    for i, (cp, lbl) in enumerate(zip(active_crops, active_labels)):
        log.append("  [%d/%d] %s …" % (i + 1, n, lbl))

        # Dict format tells image_to_audio to skip YOLO and use
        # the crop itself as the only detected region.
        # locality=1.0 means "take this object at face value"
        crop_dict = {cp: [(cp, 1.0)]}

        obj_save_dir = tracks_dir / ("obj_%d" % i)
        obj_save_dir.mkdir(exist_ok=True)

        i2a_image_to_audio(
            crop_dict, text="", transcription="",
            save_dir=str(obj_save_dir), config=cfg_path,
            gen_remix=True, gen_tracks=False, emb_only=False,
            pretrained=ckpt_path, batch_size=bs, var_samples=vs,
            shuffle_remix=True, cycle_its=ci, cycle_samples=cs,
            keep_data_cache=True,   # dict input → no cache dir to delete
            duration=dur, seed=int(seed),
            device=DEVICE,
        )
        torch.cuda.empty_cache()

        # image_to_audio saves as <stem>.wav  (stem of the crop path)
        obj_wav = str(obj_save_dir / (Path(cp).stem + ".wav"))
        wave, _ = _sf.read(obj_wav)
        if wave.ndim > 1:
            wave = wave[:, 0]
        wave = np.array(wave, dtype=np.float32)
        per_object_waves_list.append(wave)
        log.append("    Peak=%.4f" % float(np.abs(wave).max()))

    # Stack to [N, 1, T] — pad/trim to same length
    max_len = max(w.shape[0] for w in per_object_waves_list)
    per_object_waves = np.zeros((n, 1, max_len), dtype=np.float32)
    for i, w in enumerate(per_object_waves_list):
        per_object_waves[i, 0, :w.shape[0]] = w

    # ────────────────────────────────────────────────────────────────
    # 3. Locality + pan — camera-aware if a 3D camera was applied
    # ────────────────────────────────────────────────────────────────
    objects_3d   = sess.get("objects_3d", [])
    listener_pos = sess.get("listener_pos", None)

    if objects_3d and listener_pos and len(objects_3d) >= n:
        # Camera was set via "Choose this camera" → use 3D projection
        cam_pos = np.array(listener_pos)

        # Camera forward: look at scene centroid
        positions = np.array([[o.X, o.Y, o.Z] for o in objects_3d])
        centroid  = positions.mean(axis=0)
        cam_fwd   = centroid - cam_pos
        cam_fwd  /= (np.linalg.norm(cam_fwd) + 1e-12)

        world_up  = np.array([0.0, 1.0, 0.0])
        cam_right = np.cross(cam_fwd, world_up)
        r_len     = np.linalg.norm(cam_right)
        if r_len < 1e-6:
            cam_right = np.array([1.0, 0.0, 0.0])
        else:
            cam_right /= r_len
        cam_up = np.cross(cam_right, cam_fwd)

        localities, pans = _project_3d_to_localities(
            objects_3d[:n], cam_pos, cam_fwd, cam_right, cam_up,
            float(hfov), ww, hh,
        )
        log.append("Using 3D camera position for spatial mix")
    else:
        # No 3D camera → use original YOLO bbox localities
        localities, pans = _compute_localities_and_pans(active_bboxes, ww, hh)
        log.append("Using YOLO bbox localities (no 3D camera set)")
    # Gains: normalised so biggest = 1.0, power 2 curve for strong contrast
    _max_loc = max(localities) if max(localities) > 0 else 1e-6
    gains = [(loc / _max_loc) ** 2.0 for loc in localities]

    log.append("Locality-based spatial mix:")
    for i in range(n):
        log.append("  %s: locality=%.3f  gain=%.3f (%+.1f dB)  pan=%+.2f" % (
            active_labels[i], localities[i], gains[i],
            20 * math.log10(max(gains[i], 1e-8)), pans[i]))

    # ────────────────────────────────────────────────────────────────
    # 4. Cache for re-rendering
    # ────────────────────────────────────────────────────────────────
    sess["per_object_waves"] = per_object_waves
    sess["audio_labels"]     = list(active_labels)
    sess["audio_bboxes"]     = list(active_bboxes)
    sess["localities"]       = localities
    sess["pans"]             = pans
    sess["ref_wave"]         = ref_wave
    sess["ref_sr"]           = ref_sr
    sess["ref_bbox_areas"]   = [abs(b[2]-b[0]) * abs(b[3]-b[1]) for b in active_bboxes]

    # ────────────────────────────────────────────────────────────────
    # 5. Locality stereo mix
    # ────────────────────────────────────────────────────────────────
    stereo = _locality_stereo_mix(per_object_waves, localities, pans, sr=SAMPLE_RATE)
    stereo_out = stereo.T.astype(np.float32)  # [samples, 2]

    log.append("Done ✓  (%.1fs @ %d Hz)" % (stereo_out.shape[0] / SAMPLE_RATE, SAMPLE_RATE))

    table = _locality_table_md(active_labels, localities, gains, pans)
    status = "\n\n".join(log) + "\n\n" + table

    return (SAMPLE_RATE, stereo_out), (ref_sr, ref_wave), status


# ═══════════════════════════════════════════════════════════════════════════
# Step 4b — Re-render from camera angle (locality re-projection)
# ═══════════════════════════════════════════════════════════════════════════

def step_rerender_spatial_audio(sid, cam_x, cam_y, yaw, pitch,
                                hfov, attenuation, enable_delay):
    """Re-mix cached per-object tracks using re-projected localities.

    When the camera moves in the 3D viewer:
    - Project each object's 3D position into the new camera view
    - Objects that appear bigger (closer) → higher locality → louder
    - Objects that appear more left/right → stereo panning

    If no 3D world was generated, falls back to original YOLO localities.
    No model inference — instant.
    """
    if not sid or sid not in _sessions:
        return None, None, "Run detection first."

    sess = _sessions[sid]
    if "per_object_waves" not in sess:
        return None, None, "Generate audio first, then move camera to re-render."

    per_object_waves = sess["per_object_waves"]
    ref_wave = sess["ref_wave"]
    ref_sr   = sess["ref_sr"]
    labels   = sess["audio_labels"]
    ww, hh   = sess["img_width"], sess["img_height"]
    hfov_val = float(hfov)
    n = per_object_waves.shape[0]

    log = ["Re-rendering from camera angle (%d objects) …" % n]

    objects_3d = sess.get("objects_3d", [])

    if objects_3d and len(objects_3d) >= n:
        # ── 3D world exists: re-project from new camera angle ──
        listener_pos = sess.get("listener_pos", None)

        if listener_pos is not None:
            cam_pos = np.array(listener_pos)
        else:
            # Approximate from 2D sliders
            cam_pos = np.array([float(cam_x) - ww / 2, 0.0, float(cam_y) - hh / 2])

        # Camera forward: always look at the scene centroid
        # (yaw/pitch sliders are for the 2D preview, not the 3D orbit)
        positions = np.array([[o.X, o.Y, o.Z] for o in objects_3d])
        centroid  = positions.mean(axis=0)
        cam_fwd   = centroid - cam_pos
        cam_fwd  /= (np.linalg.norm(cam_fwd) + 1e-12)

        world_up = np.array([0.0, 1.0, 0.0])
        cam_right = np.cross(cam_fwd, world_up)
        r_len = np.linalg.norm(cam_right)
        if r_len < 1e-6:
            cam_right = np.array([1.0, 0.0, 0.0])
        else:
            cam_right /= r_len
        cam_up = np.cross(cam_right, cam_fwd)

        ref_sizes = sess.get("ref_bbox_areas", None)
        orig_localities = sess.get("localities", None)

        localities, pans = _project_3d_to_localities(
            objects_3d[:n], cam_pos, cam_fwd, cam_right, cam_up,
            hfov_val, ww, hh, ref_sizes=ref_sizes,
            original_localities=orig_localities,
        )
        log.append("Re-projected from 3D positions")

    else:
        # ── No 3D world: use original YOLO localities ──
        localities = sess.get("localities", [0.1] * n)
        pans = sess.get("pans", [0.0] * n)
        log.append("Using original YOLO localities (no 3D world)")

    _max_loc = max(localities) if max(localities) > 0 else 1e-6
    gains = [(loc / _max_loc) ** 2.0 for loc in localities]

    # ── Mix ──
    stereo = _locality_stereo_mix(per_object_waves, localities, pans, sr=SAMPLE_RATE)
    stereo_out = stereo.T.astype(np.float32)

    log.append("Done ✓  (instant re-render, no model inference)")

    for i in range(n):
        log.append("  %s: locality=%.3f  gain=%.3f  pan=%+.2f" % (
            labels[i], localities[i], gains[i], pans[i]))

    table = _locality_table_md(labels, localities, gains, pans)
    status = "\n\n".join(log) + "\n\n" + table

    return (SAMPLE_RATE, stereo_out), (ref_sr, ref_wave), status


# ═══════════════════════════════════════════════════════════════════════════
# Individual mesh selector
# ═══════════════════════════════════════════════════════════════════════════

def step_select_object_mesh(sid, obj_index):
    """Return individual .glb for the selected object."""
    if not sid or sid not in _sessions:
        return None, "No session."
    sess = _sessions[sid]
    paths = sess.get("indiv_mesh_paths", [])
    labels = sess.get("labels", [])
    idx = int(obj_index)
    if idx < 0 or idx >= len(paths):
        return None, "Invalid selection."
    return paths[idx], "Showing **[%d] %s**" % (idx, labels[idx])


# ═══════════════════════════════════════════════════════════════════════════
# Build UI
# ═══════════════════════════════════════════════════════════════════════════

def build_ui():
    with gr.Blocks(
        title="SSV2A — 3D World + Spatial Audio",
        theme=gr.themes.Soft(),
        css="""
            footer { display: none !important; }
            .model3d-container { min-height: 450px; }
        """,
    ) as demo:
        gr.Markdown(
            "# SSV2A — Generative 3D World + Spatial Audio\n\n"
            "Upload an image → **YOLO detects** objects → **BLIP + SD Turbo + Hunyuan3D-2 generate 3D meshes** → "
            "positioned in a **3D world** using monocular depth → "
            "**3D spatial stereo audio** from your chosen camera angle.\n\n"
            "Each detected object becomes a **complete 3D model** you can view from any direction — "
            "see the **back, tail, body** of animals, the underside of objects, etc."
        )

        session_id = gr.State("")

        # ────────────────────────────────────────────────────────────────
        #  Row 1 : Input images + info
        # ────────────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(label="Upload Image", type="pil")
                with gr.Tabs():
                    with gr.TabItem("Detections + Camera"):
                        annotated_output = gr.Image(
                            label="Annotated (bboxes + camera)",
                            interactive=False,
                        )
                    with gr.TabItem("Depth Overlay"):
                        depth_output = gr.Image(
                            label="Depth Map",
                            interactive=False,
                        )

            with gr.Column(scale=3):
                detection_info = gr.Markdown(
                    "*Upload an image and click **① Detect**.*"
                )

        # ────────────────────────────────────────────────────────────────
        #  Row 2 : 3D World Viewer
        # ────────────────────────────────────────────────────────────────
        gr.Markdown("---\n### 3D World Viewer")
        gr.Markdown(
            "After generation, **rotate / zoom / pan** to explore the 3D world. "
            "Each object is a full 3D mesh generated by SAM segmentation → Hunyuan3D-2."
        )

        with gr.Row():
            with gr.Column(scale=3):
                world_3d_viewer = gr.Model3D(
                    label="3D World Scene (all objects)",
                    clear_color=[0.05, 0.05, 0.1, 1.0],
                    height=500,
                )
            with gr.Column(scale=2):
                single_object_viewer = gr.Model3D(
                    label="Individual Object Mesh",
                    clear_color=[0.9, 0.9, 0.95, 1.0],
                    height=400,
                )
                with gr.Row():
                    obj_selector = gr.Slider(
                        0, 10, value=0, step=1,
                        label="Object Index",
                        info="Select which object to view individually",
                    )
                    select_obj_btn = gr.Button("View Object", size="sm")
                obj_selector_info = gr.Markdown("")

        world_gen_info = gr.Markdown("")

        # ────────────────────────────────────────────────────────────────
        #  Row 2b : Unity-Style 3D World Viewer (Three.js)
        # ────────────────────────────────────────────────────────────────
        gr.Markdown("---\n### 🎮 Unity-Style 3D World Viewer")
        gr.Markdown(
            "**Navigate freely** through the generated 3D world. "
            "Use **Orbit** (drag to rotate) or **FPS** mode (WASD + mouse). "
            "Click **📍 Use This Camera Position** to set the listener location for audio generation."
        )

        unity_viewer_html = gr.HTML(
            value='<div style="text-align:center;color:#7da8ff;padding:80px 20px;'
                  'background:#1a1a2e;border-radius:8px;border:1px solid rgba(100,140,255,0.2);">'
                  '<p style="font-size:48px;margin-bottom:12px;">🎮</p>'
                  '<p style="font-size:16px;font-weight:600;">Generate a 3D World first</p>'
                  '<p style="font-size:13px;opacity:0.7;">Click ② Generate 3D World, '
                  'then the interactive Unity-style viewer will appear here.</p></div>',
            label="Unity-Style Viewer",
        )

        # Hidden fields to receive camera data from the Three.js iframe via JS
        unity_cam_x = gr.Number(value=0, visible=False, label="Unity Cam X", elem_id="unity_cam_x")
        unity_cam_y = gr.Number(value=0, visible=False, label="Unity Cam Y", elem_id="unity_cam_y")
        unity_cam_z = gr.Number(value=0, visible=False, label="Unity Cam Z", elem_id="unity_cam_z")
        unity_cam_yaw = gr.Number(value=0, visible=False, label="Unity Yaw", elem_id="unity_cam_yaw")
        unity_cam_pitch = gr.Number(value=0, visible=False, label="Unity Pitch", elem_id="unity_cam_pitch")

        apply_unity_cam_btn = gr.Button(
            "📍 Apply 3D Camera → Audio Sliders",
            variant="secondary", interactive=True,
        )

        # ────────────────────────────────────────────────────────────────
        #  Row 3 : Camera / Listener controls
        # ────────────────────────────────────────────────────────────────
        gr.Markdown("---\n### Camera / Listener Controls")
        gr.Markdown(
            "**Position** — where the listener sits on the image plane.  "
            "**Yaw** — head rotation left / right.  "
            "**Pitch** — head tilt up / down.  \n"
            "Objects to the **left of your look direction → left speaker**; "
            "farther objects → **quieter + more reverb**."
        )

        with gr.Row():
            slider_x = gr.Slider(
                0, 512, value=256, step=1,
                label="Camera X  (← left … right →)",
                interactive=False,
            )
            slider_y = gr.Slider(
                0, 512, value=256, step=1,
                label="Camera Y  (↑ top … bottom ↓)",
                interactive=False,
            )
            yaw_slider = gr.Slider(
                -180, 180, value=0, step=1,
                label="Yaw °  (← left … right →)",
                info="+ve = look right",
            )
            pitch_slider = gr.Slider(
                -60, 60, value=0, step=1,
                label="Pitch °  (↓ down … up ↑)",
                info="+ve = look up",
            )

        with gr.Row():
            hfov_slider = gr.Slider(
                30, 120, value=60, step=5,
                label="Horizontal FoV (°)",
                info="60° standard, 90° wide, 30° tele",
            )
            attenuation_slider = gr.Slider(
                0.3, 2.0, value=0.8, step=0.1,
                label="Distance Attenuation",
                info="0.5 gentle · 1.0 linear · 2.0 inverse-square",
            )
            delay_checkbox = gr.Checkbox(
                value=True, label="Propagation Delay",
                info="Speed-of-sound delay (343 m/s)",
            )

        listener_info = gr.Markdown("")

        # ────────────────────────────────────────────────────────────────
        #  Buttons
        # ────────────────────────────────────────────────────────────────
        with gr.Row():
            detect_btn = gr.Button("① Detect + Depth", variant="primary")
            gen_3d_btn = gr.Button(
                "② Generate 3D World",
                variant="primary", interactive=False,
            )
            preview_btn = gr.Button("③ Preview Listener", interactive=False)
            gen_audio_btn = gr.Button(
                "④ Generate 3D Spatial Audio",
                variant="primary", interactive=False,
            )
            rerender_btn = gr.Button(
                "🔄 Re-render (camera move)",
                variant="secondary", interactive=True,
            )

        # ────────────────────────────────────────────────────────────────
        #  Generation settings (collapsible)
        # ────────────────────────────────────────────────────────────────
        with gr.Accordion("Pipeline & Generation Settings", open=False):
            with gr.Row():
                pipeline_choice = gr.Radio(
                    ["i2a (original)", "mm2a (LLM filter + BLIP)"],
                    value="i2a (original)", label="Pipeline",
                )
                depth_model_choice = gr.Dropdown(
                    ["depth-pro",
                     "zoedepth-nyu-kitti", "zoedepth-nyu",
                     "depth-anything-v2-large",
                     "depth-anything-v2-base",
                     "depth-anything-v2-small",
                     "dpt-large"],
                    value="depth-pro",
                    label="Depth Model",
                )
                mesh_resolution = gr.Slider(
                    20, 75, value=50, step=5,
                    label="Hunyuan3D-2 Steps",
                    info="50 = best quality (guidance=7.5), 30 = faster",
                )
            with gr.Row():
                cfg_input  = gr.Textbox(value=DEFAULT_CFG, label="Config JSON")
                ckpt_input = gr.Textbox(value=DEFAULT_CKPT, label="Checkpoint")
            with gr.Row():
                bs_sl       = gr.Slider(1, 64, value=16, step=1, label="Batch Size")
                vs_sl       = gr.Slider(1, 128, value=64, step=1, label="Var Samples")
                ci_sl       = gr.Slider(1, 128, value=64, step=1, label="Cycle Its")
                cs_sl       = gr.Slider(1, 128, value=64, step=1, label="Cycle Samples")
            with gr.Row():
                dur_sl      = gr.Slider(1, 30, value=10, step=1, label="Duration (s)")
                seed_input  = gr.Number(value=42, label="Seed", precision=0)

        gr.Markdown("---\n### Audio Output")
        with gr.Row():
            spatial_audio = gr.Audio(label="🎧  3D Spatial Stereo Mix")
            mono_audio    = gr.Audio(label="Mono Remix (reference)")
        gen_status = gr.Markdown("")

        # ── Examples ──
        ex_dir = Path("images")
        if ex_dir.exists():
            imgs = sorted([str(p) for p in ex_dir.iterdir()
                           if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])[:8]
            if imgs:
                gr.Markdown("### Example Images")
                gr.Examples(examples=[[i] for i in imgs], inputs=[image_input])

        # ────────────────────────────────────────────────────────────────
        #  Wiring
        # ────────────────────────────────────────────────────────────────

        # ① Detect + Depth
        detect_btn.click(
            fn=step_detect_depth,
            inputs=[image_input, cfg_input, ckpt_input, bs_sl,
                    depth_model_choice, hfov_slider],
            outputs=[
                annotated_output, depth_output,
                detection_info,
                slider_x, slider_y,
                gen_3d_btn,    # enable after detection
                gen_audio_btn, # enable (they can skip 3D gen for audio only)
                session_id,
            ],
        )

        # ② Generate 3D World
        gen_3d_btn.click(
            fn=step_generate_3d_world,
            inputs=[session_id, mesh_resolution, hfov_slider],
            outputs=[world_3d_viewer, single_object_viewer, world_gen_info,
                     unity_viewer_html],
        )

        # ②½ Apply Unity Camera → Audio Sliders
        apply_unity_cam_btn.click(
            fn=step_apply_unity_camera,
            inputs=[session_id,
                    unity_cam_x, unity_cam_y, unity_cam_z,
                    unity_cam_yaw, unity_cam_pitch,
                    hfov_slider],
            outputs=[slider_x, slider_y, yaw_slider, pitch_slider,
                     listener_info],
        )

        # ③ Preview Listener
        preview_btn.click(
            fn=step_preview_listener,
            inputs=[session_id, slider_x, slider_y,
                    yaw_slider, pitch_slider,
                    hfov_slider, attenuation_slider, delay_checkbox],
            outputs=[annotated_output, listener_info],
        )

        # Live preview on slider release
        cam_sliders = [slider_x, slider_y, yaw_slider, pitch_slider]
        for sl in cam_sliders:
            sl.release(
                fn=step_preview_listener,
                inputs=[session_id, slider_x, slider_y,
                        yaw_slider, pitch_slider,
                        hfov_slider, attenuation_slider, delay_checkbox],
                outputs=[annotated_output, listener_info],
            )

        # Individual object selector
        select_obj_btn.click(
            fn=step_select_object_mesh,
            inputs=[session_id, obj_selector],
            outputs=[single_object_viewer, obj_selector_info],
        )

        # ④ Generate Audio
        gen_audio_btn.click(
            fn=step_generate_audio,
            inputs=[
                session_id, pipeline_choice, cfg_input, ckpt_input,
                bs_sl, vs_sl, ci_sl, cs_sl,
                dur_sl, seed_input,
                slider_x, slider_y, yaw_slider, pitch_slider,
                hfov_slider, attenuation_slider, delay_checkbox,
            ],
            outputs=[spatial_audio, mono_audio, gen_status],
        )

        # ④b Re-render spatial audio (instant, no model inference)
        rerender_btn.click(
            fn=step_rerender_spatial_audio,
            inputs=[
                session_id,
                slider_x, slider_y, yaw_slider, pitch_slider,
                hfov_slider, attenuation_slider, delay_checkbox,
            ],
            outputs=[spatial_audio, mono_audio, gen_status],
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SSV2A — Generative 3D World + Spatial Audio — Gradio UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    # ── Build a FastAPI wrapper with a custom route for serving viewer HTML ──
    # Gradio's file server blocks .html for security.  We add our own route
    # that serves *only* viewer.html files from /tmp/3dworld_* directories.
    from fastapi import FastAPI as _FastAPI
    from fastapi.responses import HTMLResponse as _HTMLResponse
    import uvicorn as _uvicorn

    app = _FastAPI()

    @app.get("/viewer_page")
    async def _serve_viewer_page(path: str = ""):
        """Serve a viewer.html file generated during 3D-world builds."""
        # Security: only serve viewer.html inside /tmp/3dworld_*
        if (not path.startswith("/tmp/3dworld_")
                or not path.endswith("/viewer.html")
                or ".." in path):
            return _HTMLResponse("Forbidden", status_code=403)
        try:
            with open(path, "r") as fh:
                return _HTMLResponse(fh.read())
        except FileNotFoundError:
            return _HTMLResponse("Viewer file not found", status_code=404)

    from fastapi.responses import Response as _Response

    @app.get("/viewer_asset")
    async def _serve_viewer_asset(path: str = ""):
        """Serve GLB (or other assets) from /tmp/3dworld_* dirs."""
        if (not path.startswith("/tmp/3dworld_")
                or ".." in path):
            return _Response("Forbidden", status_code=403)
        # Only allow known safe extensions
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

    # Build the Gradio UI and mount it at /
    demo = build_ui()
    demo.queue()   # required: enables /queue/status — without this the JS hangs on a white screen
    app = gr.mount_gradio_app(app, demo, path="/")

    print(f"\n  Launching on http://0.0.0.0:{args.port}  (viewer route at /viewer_page)\n")
    _uvicorn.run(app, host="0.0.0.0", port=args.port)
