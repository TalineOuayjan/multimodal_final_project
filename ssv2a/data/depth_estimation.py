"""
depth_estimation.py — Monocular depth estimation for 3D spatial audio.

Uses a pretrained monocular depth model to produce a per-pixel depth map
from a single RGB image.

Public API
----------
    load_depth_model(model_name, device)  → (model, processor)
    estimate_depth(image_path, model, processor, device, model_name) → depth_map [H, W]
    depth_for_bboxes(depth_map, bboxes)   → list[float]  (median depth per object)

Supported models (pass to ``model_name``):
    "depth-anything-v2-small"   — fast, decent quality        (default)
    "depth-anything-v2-base"    — balanced
    "depth-anything-v2-large"   — best quality, slower
    "depth-pro"                 — Apple Depth Pro — metric depth, sharp edges   ★★★
    \"zoedepth-nyu-kitti\"           — ZoeDepth NK    — metric depth from NYU+KITTI ★★
    \"zoedepth-nyu\"                — ZoeDepth N     — metric depth, NYU indoor
    "dpt-large"                 — Intel DPT (older but solid)

Output convention:
    Higher depth = farther from camera (pseudo-metric, max ≈ 100).
    Models that output metric depth (Depth Pro, ZoeDepth) are re-scaled to
    the same [0..100] range for consistency with the spatial audio pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from PIL import Image

# ── Model registry ──────────────────────────────────────────────────────────

_MODEL_HUB_IDS = {
    # ── Depth Anything V2 (relative / inverse depth) ──
    "depth-anything-v2-small": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything-v2-base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything-v2-large": "depth-anything/Depth-Anything-V2-Large-hf",
    # ── Apple Depth Pro (metric depth — sharp edges, best quality) ──
    "depth-pro":               "apple/DepthPro-hf",
    # ── ZoeDepth (metric depth — trained on NYU/KITTI) ──
    "zoedepth-nyu-kitti":      "Intel/zoedepth-nyu-kitti",
    "zoedepth-nyu":            "Intel/zoedepth-nyu",
    # ── Intel DPT (older, inverse depth) ──
    "dpt-large":               "Intel/dpt-large",
}

# Models that output metric depth directly (metres, higher = farther).
# All others output inverse / relative depth (needs 1/d conversion).
_METRIC_MODELS = {"depth-pro", "zoedepth-nyu-kitti", "zoedepth-nyu"}

DEFAULT_MODEL = "depth-anything-v2-small"

# ── HF cache directory ──────────────────────────────────────────────────────
# Default to /Data/<user>/hf_cache when running on a machine where the home
# directory has a tight disk quota (common on university clusters).

def _default_hf_cache() -> Optional[str]:
    """Return a cache directory on /Data if it exists and is writable."""
    data_root = Path("/Data") / os.getenv("USER", "")
    candidate = data_root / "hf_cache"
    if candidate.is_dir() and os.access(candidate, os.W_OK):
        return str(candidate)
    return None


# ── Load model ──────────────────────────────────────────────────────────────

def load_depth_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
):
    """Load a monocular depth estimation model + image processor.

    Returns
    -------
    model : PreTrainedModel
    processor : AutoImageProcessor
    """
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor

    hub_id = _MODEL_HUB_IDS.get(model_name, model_name)
    print(f"[DEPTH] Loading depth model: {hub_id} …")

    # Fall back to /Data HF cache when no explicit cache_dir is given
    if cache_dir is None:
        cache_dir = _default_hf_cache()

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
        print(f"[DEPTH] HF cache dir → {cache_dir}")

    processor = AutoImageProcessor.from_pretrained(hub_id, **kwargs)
    model = AutoModelForDepthEstimation.from_pretrained(
        hub_id, torch_dtype=torch.float32, **kwargs,
    )
    model = model.to(device).eval()

    print(f"[DEPTH] Depth model ready on {device}."
          f"  (metric={model_name in _METRIC_MODELS})")
    return model, processor


# ── Estimate depth map ──────────────────────────────────────────────────────

@torch.no_grad()
def estimate_depth(
    image_path: str,
    model,
    processor,
    device: str = "cuda",
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Produce a dense depth map for a single image.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    model : PreTrainedModel
        Depth estimation model from ``load_depth_model``.
    processor : AutoImageProcessor
        Corresponding image processor.
    device : str
        Computation device.
    model_name : str
        Key into the model registry (needed to decide metric vs inverse).

    Returns
    -------
    depth_map : np.ndarray, shape [H, W], dtype float32
        **Pseudo-metric depth** (higher = farther from camera), max ≈ 100.
    """
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # predicted_depth: [1, Hm, Wm] or [Hm, Wm]
    predicted_depth = outputs.predicted_depth
    if predicted_depth.dim() == 2:
        predicted_depth = predicted_depth.unsqueeze(0)

    # Interpolate to original image size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(orig_h, orig_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()  # [H, W]

    raw = prediction.cpu().numpy().astype(np.float32)

    is_metric = model_name in _METRIC_MODELS

    if is_metric:
        # ── Metric models (Depth Pro, ZoeDepth) ──
        # Output is depth in metres — higher = farther.  Clamp negatives.
        depth_map = np.clip(raw, 0.0, None)
    else:
        # ── Inverse / relative depth models (Depth Anything, DPT) ──
        # Output: higher value = *closer* (MiDaS convention).
        # Convert to depth: Z = 1 / (normalised_inverse + ε)
        d_min, d_max = raw.min(), raw.max()
        if d_max - d_min > 1e-6:
            inv_norm = (raw - d_min) / (d_max - d_min)
        else:
            inv_norm = np.zeros_like(raw)
        depth_map = 1.0 / (inv_norm + 0.01)

    # Re-scale so max depth ≈ 100 (consistent across all models)
    mx = depth_map.max()
    if mx > 1e-8:
        depth_map = depth_map / mx * 100.0

    return depth_map


# ── Extract per-object depth from bounding boxes ───────────────────────────

def depth_for_bboxes(
    depth_map: np.ndarray,
    bboxes: List[List[float]],
    method: str = "median",
) -> List[float]:
    """Extract a representative depth value for each bounding box.

    Parameters
    ----------
    depth_map : np.ndarray [H, W]
        Dense depth map from ``estimate_depth``.
    bboxes : list of [x1, y1, x2, y2]
        Bounding boxes in pixel coordinates.
    method : str
        Aggregation method: 'median' (robust) or 'mean'.

    Returns
    -------
    depths : list[float]
        One depth value per bounding box (higher = farther).
    """
    H, W = depth_map.shape
    depths = []
    agg = np.median if method == "median" else np.mean

    for box in bboxes:
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        # Clamp to image bounds
        x1 = max(0, min(x1, W - 1))
        x2 = max(x1 + 1, min(x2, W))
        y1 = max(0, min(y1, H - 1))
        y2 = max(y1 + 1, min(y2, H))

        # Use centre 60% of bbox to avoid edge noise
        cx_range = int((x2 - x1) * 0.2)
        cy_range = int((y2 - y1) * 0.2)
        region = depth_map[y1 + cy_range : y2 - cy_range,
                           x1 + cx_range : x2 - cx_range]

        if region.size == 0:
            region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            depths.append(1.0)  # fallback: very close
        else:
            depths.append(float(agg(region)))

    return depths


# ── Visualise depth map (optional, for debugging) ──────────────────────────

def save_depth_visualisation(
    depth_map: np.ndarray,
    save_path: str,
    colormap: str = "magma",
):
    """Save a colour-mapped depth map image for visual inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, figsize=(10, 8))
    im = ax.imshow(depth_map, cmap=colormap)
    ax.set_title("Estimated Depth (brighter = farther)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DEPTH] Depth visualisation saved → {save_path}")
