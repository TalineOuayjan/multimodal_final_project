#!/usr/bin/env python3
"""
hy3d_worker.py — Hunyuan3D-2 full pipeline: shape + PBR texture.

Called as a subprocess from the main pipeline (ssv2a_env / Python 3.9)
with the hy3d_env (Python 3.10).

Usage:
    python hy3d_worker.py --image /path/to/input.png --output /path/to/output.glb
                          [--steps 30] [--device cuda:0] [--no-texture]

Full pipeline:
  1. Background removal (rembg)
  2. Hunyuan3D-2 DiT shape generation (600K+ vertices)
  3. Hunyuan3D-Paint texture generation (PBR UV-mapped textures)
     - Delight (shadow/lighting removal)
     - Multi-view diffusion (6-view texture synthesis)
     - UV baking + inpainting
"""

import argparse
import json
import os
import sys
import time

# ── Cache directories — MUST be set BEFORE any HF/torch imports ───────
os.environ.setdefault("HF_HUB_CACHE", "/Data/taline.ouayjan/hf_cache/hub")
os.environ.setdefault("HF_HOME", "/Data/taline.ouayjan/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/Data/taline.ouayjan/hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/Data/taline.ouayjan/hf_cache/transformers")
os.environ.setdefault("U2NET_HOME", "/Data/taline.ouayjan/hf_cache/u2net")
os.environ.setdefault("HF_MODULES_CACHE", "/Data/taline.ouayjan/hf_cache/modules")
os.environ.setdefault("XDG_CACHE_HOME", "/Data/taline.ouayjan/xdg_cache")

import numpy as np
import torch
import trimesh
from PIL import Image

# ── Hunyuan3D-2 paths ─────────────────────────────────────────────────
_HY3D_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hunyuan3D-2")
_RASTERIZER_DIR = os.path.join(_HY3D_DIR, "hy3dgen", "texgen", "custom_rasterizer")
_RENDERER_DIR = os.path.join(_HY3D_DIR, "hy3dgen", "texgen", "differentiable_renderer")

for d in (_HY3D_DIR, _RASTERIZER_DIR, _RENDERER_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.rembg import BackgroundRemover

# ── Globals ────────────────────────────────────────────────────────────
_shape_pipeline = None
_paint_pipeline = None
_rembg = None

MODEL_ID = "tencent/Hunyuan3D-2"
MODEL_SNAPSHOT = (
    "/Data/taline.ouayjan/hf_cache/hub/models--tencent--Hunyuan3D-2"
    "/snapshots/9cd649ba6913f7a852e3286bad86bfa9a2d83dcf"
)


def load_shape_pipeline(device: str = "cuda:0"):
    global _shape_pipeline
    if _shape_pipeline is not None:
        return _shape_pipeline
    print(f"[HY3D] Loading shape model ({MODEL_ID}) ...", flush=True)
    t0 = time.time()
    _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        MODEL_ID, cache_dir="/Data/taline.ouayjan/hf_cache/hub",
    )
    _shape_pipeline.to(device)
    print(f"[HY3D] Shape model ready in {time.time()-t0:.1f}s  "
          f"| VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    return _shape_pipeline


def unload_shape_pipeline():
    global _shape_pipeline
    if _shape_pipeline is not None:
        del _shape_pipeline
        _shape_pipeline = None
        torch.cuda.empty_cache()
        print(f"[HY3D] Shape model unloaded | VRAM "
              f"{torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


def load_paint_pipeline():
    global _paint_pipeline
    if _paint_pipeline is not None:
        return _paint_pipeline
    print("[HY3D] Loading paint model (delight + multiview) ...", flush=True)
    t0 = time.time()
    from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline
    _paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(MODEL_SNAPSHOT)
    _paint_pipeline.enable_model_cpu_offload()
    print(f"[HY3D] Paint model ready in {time.time()-t0:.1f}s", flush=True)
    return _paint_pipeline


def load_rembg():
    global _rembg
    if _rembg is None:
        _rembg = BackgroundRemover()
    return _rembg


def generate(image_path: str, output_path: str,
             color_image_path: str = None,
             num_steps: int = 50, device: str = "cuda:0",
             use_texture: bool = True,
             octree_resolution: int = 384) -> dict:
    """Generate textured 3D mesh using full Hunyuan3D-2 pipeline."""
    t_total = time.time()

    # Load image
    image = Image.open(image_path).convert("RGBA")

    # Background removal
    rembg = load_rembg()
    image_nobg = rembg(image)
    print("[HY3D] Background removed", flush=True)

    # ── Step 1: Shape generation ───────────────────────────────────────
    shape_pipe = load_shape_pipeline(device)

    print(f"[HY3D] Generating shape ({num_steps} steps, octree={octree_resolution}, guidance=7.5) ...", flush=True)
    t0 = time.time()
    mesh = shape_pipe(
        image=image_nobg,
        num_inference_steps=num_steps,
        guidance_scale=7.5,
        octree_resolution=octree_resolution,
    )[0]
    gen_time = time.time() - t0
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)
    print(f"[HY3D] Shape: {n_verts} verts, {n_faces} faces "
          f"in {gen_time:.1f}s", flush=True)

    # Free VRAM for paint pipeline
    unload_shape_pipeline()

    # ── Step 2: Texture painting (PBR) ─────────────────────────────────
    tex_time = 0.0
    if use_texture:
        print("[HY3D] Generating PBR textures ...", flush=True)
        paint_pipe = load_paint_pipeline()
        t0 = time.time()
        textured_mesh = paint_pipe(mesh, image)
        tex_time = time.time() - t0
        print(f"[HY3D] Texture painted in {tex_time:.1f}s", flush=True)
        mesh = textured_mesh

    # ── Export ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mesh.export(output_path)
    file_size = os.path.getsize(output_path)

    total_time = time.time() - t_total
    result = {
        "vertices": n_verts,
        "faces": n_faces,
        "file_size_kb": file_size / 1024,
        "generation_time": gen_time,
        "texture_time": tex_time,
        "total_time": total_time,
        "output_path": output_path,
        "textured": use_texture,
        "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    print(f"[HY3D] Done: {output_path} ({file_size/1024:.0f} KB, "
          f"{total_time:.1f}s total, textured={use_texture})", flush=True)
    # JSON result on last line — parsed by subprocess caller
    print("__RESULT__" + json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hunyuan3D-2 shape + texture worker")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output .glb path")
    parser.add_argument("--color-image", default=None,
                        help="(backward compat, unused)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Shape diffusion steps (default 50)")
    parser.add_argument("--octree-resolution", type=int, default=384,
                        help="Octree resolution for mesh (default 384)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-texture", action="store_true",
                        help="Skip texture painting (shape only)")
    args = parser.parse_args()

    generate(
        image_path=args.image,
        output_path=args.output,
        num_steps=args.steps,
        device=args.device,
        use_texture=not args.no_texture,
        octree_resolution=args.octree_resolution,
    )
