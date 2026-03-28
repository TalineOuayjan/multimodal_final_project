"""
generate_3d_scene.py  --  Generative 3D world from a single image.

Pipeline:  BLIP caption  ->  SD Turbo image  ->  Hunyuan3D-2 mesh (600K+ verts)

1. YOLO detects objects  ->  bounding boxes + labels ("Lion", "Dog", ...)
2. BLIP captions each crop  ->  rich description
3. SD Turbo generates a clean 3D-ready render from the caption
4. Hunyuan3D-2 (Tencent, 1.1B params) generates a high-fidelity 3D mesh
   via subprocess (hy3d_env / Python 3.10) with front-view color projection
5. Objects placed on a Tinkercad-style grid platform
6. Scene exported as .glb

Dependencies
   transformers  (BLIP captioning)
   diffusers     (SD Turbo image generation)
   Hunyuan3D-2   (image-to-3D via hy3d_worker.py subprocess)
   trimesh, rembg, torch
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import json
import subprocess
import tempfile

import numpy as np
import torch
import trimesh
from PIL import Image

# ------------------------------------------------------------------ cache

def _default_hf_cache() -> Optional[str]:
    try:
        user = os.getenv("USER") or os.getenv("LOGNAME") or "unknown"
        candidate = Path(f"/Data/{user}/hf_cache/hub")
        if candidate.parent.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return str(candidate)
    except Exception:
        pass
    return None

def _default_transformers_cache() -> Optional[str]:
    try:
        user = os.getenv("USER") or os.getenv("LOGNAME") or "unknown"
        candidate = Path(f"/Data/{user}/hf_cache/transformers")
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    return None

def _set_cache_env() -> None:
    cache = _default_hf_cache()
    if cache and not os.environ.get("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = cache
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache
    tcache = _default_transformers_cache()
    if tcache:
        os.environ.setdefault("TRANSFORMERS_CACHE", tcache)

_set_cache_env()

# ======================================================= model singletons

_blip_processor = None
_blip_model     = None
_sd_pipe        = None
_sam_predictor  = None

# ----- SAM (Segment Anything) ----------------------------------------

def load_sam(device: str = "cuda:0", model_path: str = "sam_b.pt"):
    """Load SAM via ultralytics (reuses same weights as detect_spatial)."""
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor
    from ultralytics.models.sam import Predictor as SAMPredictor
    # Resolve model path relative to SSV2A project root
    project_root = Path(__file__).resolve().parent.parent.parent
    weight = project_root / model_path
    if not weight.exists():
        weight = Path(model_path)  # fallback to given path
    print(f"[SAM] Loading {weight} ...")
    overrides = dict(
        conf=0.25, retina_masks=True, task="segment", mode="predict",
        model=str(weight), save=False, verbose=False, device=device,
    )
    _sam_predictor = SAMPredictor(overrides=overrides)
    print("[SAM] Ready.")
    return _sam_predictor


def unload_sam():
    global _sam_predictor
    _sam_predictor = None
    torch.cuda.empty_cache()


def segment_object(image_pil: Image.Image, bbox: list,
                   device: str = "cuda:0",
                   bg_color: tuple = (255, 255, 255)) -> Image.Image:
    """Use SAM to segment the object inside *bbox* and return it
    isolated on a solid background **with alpha channel**.

    The output is a 512x512 RGBA image:
    - Object pixels retain original colour with alpha=255
    - Background is transparent (alpha=0)
    - The object is centered in a square canvas with ~20% border
      so Hunyuan3D-2's recenter() preserves correct proportions.
    """
    import tempfile
    sam = load_sam(device)

    # SAM needs a file path
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
        image_pil.save(tmp_path, "PNG")

    sam.set_image(tmp_path)
    bbox_tensor = torch.tensor([bbox], dtype=torch.float32, device=device)
    results = sam(bboxes=bbox_tensor)
    sam.reset_image()
    os.unlink(tmp_path)

    # Extract mask
    mask = results[0].masks.data.cpu().numpy()          # [1, H, W]
    mask = np.squeeze(mask, axis=0).astype(np.uint8)    # [H, W]

    img_np = np.array(image_pil.convert("RGB"))

    # Tight bounding box around the SAM mask
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        # Fallback: use the detection bbox
        x0, y0, x1, y1 = [int(v) for v in bbox]
    else:
        pad = 10
        y0 = max(0, ys.min() - pad)
        y1 = min(mask.shape[0], ys.max() + pad)
        x0 = max(0, xs.min() - pad)
        x1 = min(mask.shape[1], xs.max() + pad)

    # Crop object + mask
    crop_rgb = img_np[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]

    # Build RGBA: object pixels + alpha from SAM mask
    h, w = crop_rgb.shape[:2]
    alpha = (crop_mask * 255).astype(np.uint8)
    rgba = np.dstack([crop_rgb, alpha])  # [h, w, 4]

    # Square-pad: place object in center of a square canvas (transparent bg)
    # with ~20% border so Hunyuan3D-2 gets good centering
    side = max(h, w)
    border = int(side * 0.20)          # 20% extra border
    canvas_size = side + 2 * border
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)

    # Center the crop in the canvas
    oy = (canvas_size - h) // 2
    ox = (canvas_size - w) // 2
    canvas[oy:oy+h, ox:ox+w] = rgba

    # Resize to 512x512 (preserving aspect ratio via the square canvas)
    result = Image.fromarray(canvas, "RGBA").resize(
        (512, 512), Image.LANCZOS)
    return result


# ----- BLIP ----------------------------------------------------------

def load_blip(device: str = "cuda:0"):
    global _blip_processor, _blip_model
    if _blip_model is not None:
        return _blip_processor, _blip_model
    import warnings; warnings.filterwarnings("ignore")
    from transformers import BlipProcessor, BlipForConditionalGeneration
    cache = _default_transformers_cache() or _default_hf_cache()
    print("[BLIP] Loading Salesforce/blip-image-captioning-large ...")
    _blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large", cache_dir=cache)
    _blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        cache_dir=cache, torch_dtype=torch.float16).to(device)
    _blip_model.eval()
    print("[BLIP] Ready.")
    return _blip_processor, _blip_model

def unload_blip():
    global _blip_processor, _blip_model
    _blip_processor = None; _blip_model = None
    torch.cuda.empty_cache()

def _singularise_caption(caption: str, label: str) -> str:
    """Strip plurality cues so SD Turbo generates exactly ONE object."""
    import re
    # Remove leading number words / digits  ("two dogs" -> "dogs")
    caption = re.sub(
        r'\b(two|three|four|five|six|seven|eight|nine|ten|\d+)\s+',
        '', caption, flags=re.IGNORECASE)
    # Remove grouping phrases
    for phrase in ['a group of', 'a pair of', 'a couple of',
                   'group of', 'pair of', 'couple of',
                   'several', 'many', 'multiple', 'some']:
        caption = re.sub(re.escape(phrase), '', caption, flags=re.IGNORECASE)
    # Force article to singular
    caption = re.sub(r'^\s*a?\s*photos?\s+of\s+', '', caption, flags=re.IGNORECASE)
    caption = caption.strip(' ,.')
    # Prepend "a single {label}" to anchor the meaning
    return f"a single {label.lower()}, {caption}"


def caption_crop(image_crop: Image.Image, label: str,
                 device: str = "cuda:0") -> str:
    proc, model = load_blip(device)
    img = image_crop.convert("RGB").resize((384, 384))
    cond = f"a photo of a single {label.lower()},"
    inputs = proc(img, cond, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        ids = model.generate(**inputs, max_length=60)
    raw = proc.decode(ids[0], skip_special_tokens=True)
    clean = _singularise_caption(raw, label)
    print(f"    BLIP raw: {raw!r}  →  clean: {clean!r}")
    return clean

# ----- SD Turbo -------------------------------------------------------

def load_sd_turbo(device: str = "cuda:0"):
    global _sd_pipe
    if _sd_pipe is not None:
        return _sd_pipe
    import warnings; warnings.filterwarnings("ignore")
    from diffusers import AutoPipelineForText2Image
    cache = _default_hf_cache()
    print("[SD Turbo] Loading stabilityai/sd-turbo ...")
    _sd_pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float16,
        variant="fp16", cache_dir=cache)
    _sd_pipe.to(device)
    print("[SD Turbo] Ready.")
    return _sd_pipe

def unload_sd_turbo():
    global _sd_pipe
    _sd_pipe = None; torch.cuda.empty_cache()

def generate_clean_image(caption: str, device: str = "cuda:0",
                         num_steps: int = 4) -> Image.Image:
    pipe = load_sd_turbo(device)
    prompt = (f"{caption}, exactly one object, solo, centered, full body, "
              "3d render, studio lighting, plain gray background, "
              "high quality, detailed, one subject only")
    negative = ("multiple, two, pair, group, crowd, several, "
                "duplicate, clone, split screen, collage")
    print(f"    SD prompt: {prompt!r}")
    with torch.no_grad():
        result = pipe(prompt, num_inference_steps=num_steps,
                      guidance_scale=0.0,
                      negative_prompt=negative)
    return result.images[0]

# ----- Hunyuan3D-2 via subprocess --------------------------------------

_HY3D_ENV_PYTHON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "hy3d_env", "bin", "python",
)
_HY3D_WORKER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "hy3d_worker.py",
)

def _run_hunyuan3d(image_path: str, output_path: str,
                   color_image_path: str = None,
                   num_steps: int = 50,
                   octree_resolution: int = 384,
                   device: str = "cuda:0") -> dict:
    """Run Hunyuan3D-2 (shape + PBR texture) as subprocess in hy3d_env."""
    cmd = [
        _HY3D_ENV_PYTHON, _HY3D_WORKER,
        "--image", image_path,
        "--output", output_path,
        "--steps", str(num_steps),
        "--octree-resolution", str(octree_resolution),
        "--device", device,
    ]

    env = os.environ.copy()
    env["HF_HUB_CACHE"] = os.environ.get(
        "HF_HUB_CACHE", "/Data/taline.ouayjan/hf_cache/hub")
    env["HF_HOME"] = os.environ.get(
        "HF_HOME", "/Data/taline.ouayjan/hf_cache")
    env["U2NET_HOME"] = os.environ.get(
        "U2NET_HOME", "/Data/taline.ouayjan/hf_cache/u2net")
    env["HF_MODULES_CACHE"] = os.environ.get(
        "HF_MODULES_CACHE", "/Data/taline.ouayjan/hf_cache/modules")
    env["HUGGINGFACE_HUB_CACHE"] = env["HF_HUB_CACHE"]
    env["XDG_CACHE_HOME"] = os.environ.get(
        "XDG_CACHE_HOME", "/Data/taline.ouayjan/xdg_cache")
    env["CUDA_HOME"] = "/usr/local/cuda-12.6.3"

    print(f"[HY3D] Launching Hunyuan3D-2 subprocess (shape + texture) ...",
          flush=True)
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=1800,
    )
    if proc.returncode != 0:
        print(f"[HY3D] STDERR: {proc.stderr[-800:]}", flush=True)
        raise RuntimeError(f"Hunyuan3D-2 failed: {proc.stderr[-400:]}")

    # Parse result from last line
    for line in reversed(proc.stdout.strip().split("\n")):
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    raise RuntimeError("Hunyuan3D-2 returned no result JSON")

# ========================= high-level: label+crop -> 3D mesh ==========

def generate_object_mesh(label: str,
                         image_crop: Optional[Image.Image] = None,
                         device: str = "cuda:0",
                         resolution: int = 256) -> trimesh.Trimesh:
    """
    Pipeline:  SAM-isolated crop  →  Hunyuan3D-2 (shape + PBR texture)

    The SAM-segmented image (single object on white bg) is fed *directly*
    to Hunyuan3D-2.  Hunyuan3D-2's built-in rembg handles background
    removal automatically, so a clean SAM crop is ideal input.

    BLIP + SD Turbo are **skipped** — they were degrading quality by
    generating flat 512x512 renders that lost 3D information.
    """
    import tempfile

    if image_crop is None:
        raise ValueError("image_crop is required (SAM-isolated object)")

    # Free any lingering BLIP/SD models from VRAM
    unload_blip()
    unload_sd_turbo()
    torch.cuda.empty_cache()

    # Feed the SAM-isolated crop directly to Hunyuan3D-2
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_img_path = os.path.join(tmp_dir, "sam_crop.png")
        mesh_out_path = os.path.join(tmp_dir, "mesh.glb")
        # Save as high-quality PNG (SAM crop is already 512x512)
        image_crop.save(input_img_path, "PNG")
        print(f"    Sending SAM crop ({image_crop.size}) → Hunyuan3D-2")

        result = _run_hunyuan3d(
            image_path=input_img_path,
            output_path=mesh_out_path,
            num_steps=resolution,  # slider-controlled (default 30)
            device=device,
        )

        # Load the generated mesh (textured GLB with materials)
        mesh = trimesh.load(mesh_out_path, force="mesh")
        tex_info = (f", texture={result['texture_time']:.0f}s"
                    if result.get('texture_time', 0) > 0 else "")
        print(f"    Hunyuan3D-2: {result['vertices']} verts, "
              f"{result['faces']} faces "
              f"(shape={result['generation_time']:.0f}s{tex_info})")

    return mesh

def _boost_vertex_colors(mesh: trimesh.Trimesh,
                          brightness: float = 1.3,
                          saturation: float = 1.1) -> None:
    if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
        return
    colors = np.array(mesh.visual.vertex_colors, dtype=np.float32)
    rgb = colors[:, :3] / 255.0
    rgb = np.clip(rgb * brightness, 0.0, 1.0)
    gray = rgb.mean(axis=1, keepdims=True)
    rgb = np.clip(gray + saturation * (rgb - gray), 0.0, 1.0)
    colors[:, :3] = (rgb * 255.0).astype(np.uint8)
    mesh.visual.vertex_colors = colors.astype(np.uint8)

# ========================= grid platform ==============================

def create_grid_platform(extent_x: float = 60.0, extent_z: float = 80.0,
                         y_level: float = 0.0, grid_spacing: float = 5.0,
                         base_color: tuple = (170, 220, 240),
                         line_color: tuple = (210, 240, 250),
                         grid_res: int = 80) -> trimesh.Trimesh:
    xs = np.linspace(-extent_x, extent_x, grid_res + 1)
    zs = np.linspace(-extent_z / 2, extent_z, grid_res + 1)
    xx, zz = np.meshgrid(xs, zs)
    yy = np.full_like(xx, y_level)
    verts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    n = len(verts)
    bc = np.array(base_color, dtype=np.uint8)
    lc = np.array(line_color, dtype=np.uint8)
    colors_rgb = np.tile(bc, (n, 1))
    lh = grid_spacing * 0.08
    for vi in range(n):
        xm = abs(verts[vi, 0]) % grid_spacing
        zm = abs(verts[vi, 2]) % grid_spacing
        if min(xm, grid_spacing - xm) < lh or min(zm, grid_spacing - zm) < lh:
            colors_rgb[vi] = lc
    colors_rgba = np.column_stack([colors_rgb, np.full(n, 255, dtype=np.uint8)])
    h_g, w_g = grid_res + 1, grid_res + 1
    faces = []
    for r in range(grid_res):
        for c in range(grid_res):
            i00 = r * w_g + c; i01 = i00 + 1
            i10 = (r+1) * w_g + c; i11 = i10 + 1
            faces.append([i00, i10, i01])
            faces.append([i01, i10, i11])
    return trimesh.Trimesh(vertices=verts,
                           faces=np.array(faces, dtype=np.int64),
                           vertex_colors=colors_rgba, process=False)

# ========================= build 3D world =============================

def _estimate_mesh_scale(bbox, depth, fx, fy):
    x1, y1, x2, y2 = bbox
    return max(abs(x2-x1)*depth/fx, abs(y2-y1)*depth/fy)

def _resolve_overlaps(positions, half_widths, iterations=10):
    """Push apart objects whose bounding extents overlap on the X axis.

    Works iteratively: for each pair that overlaps, nudge them apart
    along the X axis proportional to the overlap amount.  Preserves
    the original ordering (left stays left).

    Parameters
    ----------
    positions : np.ndarray [N, 3]  — mutable, modified in-place
    half_widths : list[float]      — half of each mesh's X extent after scaling
    iterations : int
    """
    n = len(positions)
    if n < 2:
        return
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                xi, xj = positions[i, 0], positions[j, 0]
                gap_needed = half_widths[i] + half_widths[j]
                actual_gap = abs(xj - xi)
                if actual_gap < gap_needed:
                    overlap = gap_needed - actual_gap + 0.5  # 0.5 padding
                    direction = 1.0 if xj >= xi else -1.0
                    positions[i, 0] -= direction * overlap * 0.5
                    positions[j, 0] += direction * overlap * 0.5
                    moved = True
        if not moved:
            break


def build_3d_world(meshes: List[trimesh.Trimesh], objects_3d: list,
                   camera_fx: float = 500.0, camera_fy: float = 500.0,
                   platform: Optional[trimesh.Trimesh] = None,
                   boost_colors: bool = True) -> trimesh.Scene:
    scene = trimesh.Scene()
    y_max = max(o.Y for o in objects_3d) if objects_3d else 10.0
    ground_y = y_max + 2.0
    if platform is not None:
        plat = platform.copy()
        plat.vertices[:, 1] += ground_y
        scene.add_geometry(plat, geom_name="platform")

    # ── Pre-compute positions & scales, then resolve overlaps ──
    n = len(meshes)
    scales = []
    positions = np.zeros((n, 3))
    for i, (mesh, obj) in enumerate(zip(meshes, objects_3d)):
        s = _estimate_mesh_scale(obj.bbox, obj.depth_value,
                                 camera_fx, camera_fy)
        s = max(0.5, min(s, 50.0))
        scales.append(s)
        positions[i] = [obj.X, obj.Y, obj.Z]

    # Estimate each mesh's half-width in world units after scaling
    half_widths = []
    for i, mesh in enumerate(meshes):
        centered = mesh.vertices - mesh.vertices.mean(axis=0)
        x_extent = (centered[:, 0].max() - centered[:, 0].min()) * scales[i]
        half_widths.append(x_extent * 0.5)

    _resolve_overlaps(positions, half_widths)

    # Also update objects_3d X coords so audio spatial matches visual
    for i, obj in enumerate(objects_3d):
        obj.X = float(positions[i, 0])

    for i, (mesh, obj) in enumerate(zip(meshes, objects_3d)):
        mc = mesh.copy()
        if boost_colors:
            _boost_vertex_colors(mc, brightness=1.3, saturation=1.1)
        mc.vertices -= mc.vertices.mean(axis=0)
        mc.vertices *= scales[i]
        mc.vertices += positions[i]
        scene.add_geometry(mc, geom_name=f"{i:02d}_{obj.label}")

    cam_m = trimesh.creation.icosphere(radius=0.3, subdivisions=1)
    cam_m.visual.vertex_colors = [255, 0, 0, 200]
    scene.add_geometry(cam_m, geom_name="camera_origin")
    return scene

# ========================= export =====================================

def export_scene_glb(scene: trimesh.Scene, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out = trimesh.Scene()
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            out.add_geometry(geom, geom_name=name)
    out.export(output_path)
    return os.path.abspath(output_path)

def export_individual_glbs(meshes, labels, output_dir) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, (mesh, label) in enumerate(zip(meshes, labels)):
        fp = os.path.join(output_dir, f"{i:02d}_{label.replace(' ','_')}.glb")
        mesh.export(fp); paths.append(fp)
    return paths


def export_scene_json(objects_3d: list, meshes: list,
                      scene_glb_path: str, indiv_glb_paths: list,
                      output_path: str,
                      camera_fx: float = 500.0, camera_fy: float = 500.0) -> str:
    """Export a scene.json with object positions, labels, and mesh paths.

    This JSON is consumed by the Three.js unity_viewer.html to recreate
    the 3D world in the browser with interactive camera controls.
    """
    import json as _json

    objects = []
    for i, (obj, mesh) in enumerate(zip(objects_3d, meshes)):
        scale = _estimate_mesh_scale(obj.bbox, obj.depth_value,
                                     camera_fx, camera_fy)
        scale = max(0.5, min(scale, 50.0))

        entry = {
            "index": i,
            "label": obj.label,
            "X": float(obj.X),
            "Y": float(obj.Y),
            "Z": float(obj.Z),
            "scale": float(scale),
            "depth": float(obj.depth_value),
            "distance": float(obj.distance),
            "azimuth_deg": float(obj.azimuth_deg),
            "elevation_deg": float(obj.elevation_deg),
            "bbox": [float(b) for b in obj.bbox],
            "mesh_path": indiv_glb_paths[i] if i < len(indiv_glb_paths) else None,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
        }
        objects.append(entry)

    data = {
        "scene_glb": scene_glb_path,
        "num_objects": len(objects),
        "objects": objects,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        _json.dump(data, f, indent=2)
    return os.path.abspath(output_path)

# ========================= full pipeline ==============================

def generate_3d_world_from_detections(
    image: Image.Image, bboxes: List[List[float]],
    labels: List[str], depths: List[float],
    img_width: int, img_height: int,
    device: str = "cuda:0", hfov_deg: float = 60.0,
    mesh_resolution: int = 256,
    output_path: Optional[str] = None,
) -> Tuple[trimesh.Scene, List[trimesh.Trimesh], list]:
    from ssv2a.model.spatial_3d import CameraIntrinsics, backproject_objects
    cam = CameraIntrinsics.from_image_size(img_width, img_height, hfov_deg)
    objects_3d = backproject_objects(bboxes, labels, depths,
                                    img_width, img_height,
                                    camera=cam, hfov_deg=hfov_deg)
    meshes = []
    for i, (obj, bbox) in enumerate(zip(objects_3d, bboxes)):
        print(f"[3D] Object {i+1}/{len(objects_3d)}: {obj.label}")
        x1, y1, x2, y2 = bbox
        w_b, h_b = x2-x1, y2-y1; pad = 0.3; W, H = image.size
        crop = image.crop((max(0,int(x1-w_b*pad)), max(0,int(y1-h_b*pad)),
                           min(W,int(x2+w_b*pad)), min(H,int(y2+h_b*pad))))
        mesh = generate_object_mesh(obj.label, image_crop=crop,
                                    device=device, resolution=mesh_resolution)
        meshes.append(mesh)
        print(f"  -> {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    max_z = max(o.Z for o in objects_3d) if objects_3d else 50
    max_x = max(abs(o.X) for o in objects_3d) if objects_3d else 30
    platform = create_grid_platform(extent_x=max(max_x*2, 60),
                                    extent_z=max(max_z*1.5, 80))
    scene = build_3d_world(meshes, objects_3d, camera_fx=cam.fx,
                           camera_fy=cam.fy, platform=platform,
                           boost_colors=True)
    if output_path:
        export_scene_glb(scene, output_path)
    return scene, meshes, objects_3d
