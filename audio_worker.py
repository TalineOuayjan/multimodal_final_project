#!/usr/bin/env python3
"""
audio_worker.py — SSV2A audio generation in an isolated subprocess.

Called from pipeline.py so that CUDA context corruption from the
Hunyuan3D-2 subprocess (which uses os._exit) cannot affect audio ops.

Usage:
    python audio_worker.py --params-json /tmp/params.json
                           --result-json  /tmp/result.json
"""

import argparse
import json
import os
import sys

# ── Cache dirs — set BEFORE any HF/torch imports ─────────────────────
os.environ.setdefault("HF_HUB_CACHE",         "/Data/taline.ouayjan/hf_cache/hub")
os.environ.setdefault("HF_HOME",              "/Data/taline.ouayjan/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/Data/taline.ouayjan/hf_cache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE",   "/Data/taline.ouayjan/hf_cache/transformers")
os.environ.setdefault("XDG_CACHE_HOME",       "/Data/taline.ouayjan/xdg_cache")

# numpy 2.x compat
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

# Add SSV2A project root to path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ssv2a.model.pipeline import image_to_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-json", required=True,
                        help="Path to JSON file containing audio params")
    parser.add_argument("--result-json", required=True,
                        help="Path to write result JSON")
    args = parser.parse_args()

    with open(args.params_json, "r") as f:
        p = json.load(f)

    caption = p.get("caption", "")

    if caption:
        # Combined image CLIP + text CLIP (BLIP caption) → remixer → AudioLDM
        # Matches the evaluated "Gemini-BLIP" approach from pipeline_mm2a_SAM
        import copy, json as _json
        import soundfile as sf
        import torch
        from ssv2a.data.utils import clip_embed_images, emb2seq, set_seed
        from ssv2a.model.pipeline_mm2a_SAM import clip_embed_texts_l14
        from ssv2a.model.pipeline import Pipeline
        from ssv2a.model.aldm import build_audioldm, emb_to_audio

        device      = p.get("device", "cuda")
        batch_size  = int(p.get("batch_size", 64))
        var_samples = int(p.get("var_samples", 1))
        cycle_its   = int(p.get("cycle_its", 3))
        cycle_samples = int(p.get("cycle_samples", 16))
        duration    = int(p.get("duration", 10))
        set_seed(int(p.get("seed", 42)))

        images = p["images"]  # list with one crop path

        print(f"[AUDIO_WORKER] Combined image+text CLIP | caption={caption!r} | save_dir={p['save_dir']}", flush=True)

        with open(p["config"], "r") as fp:
            config = _json.load(fp)

        # Image CLIP (SAM crop) + Text CLIP (BLIP caption)
        img_clips  = clip_embed_images(images, batch_size=batch_size, device=device)   # [1, 768]
        text_clips = clip_embed_texts_l14([caption], batch_size=batch_size, device=device)  # [1, 768]

        # Combined: [img_clip, text_clip] — 2 sources for 1 object
        combined_clips = torch.stack([img_clips[0], text_clips[0]])  # [2, 768]
        combined_jumps = [2]

        model = Pipeline(copy.deepcopy(config), p["pretrained"], device)
        model.eval()
        with torch.no_grad():
            remix_clips = emb2seq(
                combined_jumps, combined_clips,
                max_length=model.remixer.slot, delay=1, device=model.device,
            )
            remix_clips[:, 0, :] = img_clips[0]  # global = image clip
            remix_clap = model.cycle_mix(
                remix_clips, its=cycle_its, var_samples=var_samples,
                samples=cycle_samples, shuffle=p.get("shuffle_remix", True),
            )
        del model, img_clips, text_clips, combined_clips, remix_clips

        aldm = build_audioldm(model_name=config["audioldm_version"], device=device)
        waveform = emb_to_audio(aldm, remix_clap, batchsize=batch_size, duration=duration)
        sf.write(p["save_dir"], waveform[0, 0], samplerate=16000)

    else:
        images = p["images"]
        # JSON dict values are list-of-lists (tuples serialise as lists) — restore tuples
        if isinstance(images, dict):
            images = {k: [tuple(v) for v in vlist] for k, vlist in images.items()}

        print(f"[AUDIO_WORKER] Starting image_to_audio | save_dir={p['save_dir']}", flush=True)

        image_to_audio(
            images,
            text=p.get("text", ""),
            transcription=p.get("transcription", ""),
            save_dir=p["save_dir"],
            config=p["config"],
            gen_remix=p.get("gen_remix", True),
            gen_tracks=p.get("gen_tracks", False),
            emb_only=p.get("emb_only", False),
            pretrained=p["pretrained"],
            batch_size=int(p.get("batch_size", 64)),
            var_samples=int(p.get("var_samples", 1)),
            shuffle_remix=p.get("shuffle_remix", True),
            cycle_its=int(p.get("cycle_its", 3)),
            cycle_samples=int(p.get("cycle_samples", 16)),
            keep_data_cache=p.get("keep_data_cache", False),
            duration=int(p.get("duration", 10)),
            seed=int(p.get("seed", 42)),
            device=p.get("device", "cuda"),
        )

    result = {"status": "ok", "save_dir": p["save_dir"]}
    with open(args.result_json, "w") as f:
        json.dump(result, f)
    print(f"[AUDIO_WORKER] Done. Result written to {args.result_json}", flush=True)

    # Flush then hard-exit — avoids PyTorch atexit cudaDeviceReset()
    # corrupting the parent's CUDA primary context.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
