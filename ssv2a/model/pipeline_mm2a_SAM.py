"""
pipeline_mm2a_SAM.py — SSV2A multimodal pipeline: LLM filter + SAM + BLIP text conditioning.

Flow:
  1. YOLO detection (returns bounding boxes + class labels)
    2. OpenAI GPT-4o sound-source filter (keep/drop silent objects)
  3. SAM segmentation on KEPT crops using bbox + center-point prompt
  4. BLIP captions each SAM-masked image
  5. CLIP-embed SAM-masked images AND BLIP captions (both → 768-dim CLIP space)
  6. Combine image + text CLIPs as remix sources
  7. Manifold → Cycle-Mix → AudioLDM → waveform
"""

import copy
import gc
import json
import os
import os.path
import time
from pathlib import Path
from shutil import rmtree, copy2

import clip as clip_module
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from ssv2a.data.detect_gemini_SAM import detect
from ssv2a.data.tpairs import tpairs2tclips
from ssv2a.data.utils import (
    batch_extract_frames,
    clip_embed_images,
    emb2seq,
    get_timestamp,
    prior_embed_texts,
    save_wave,
    set_seed,
)
from ssv2a.model.aggregator import Aggregator
from ssv2a.model.aldm import build_audioldm, emb_to_audio
from ssv2a.model.clap import clap_embed_auds
from ssv2a.model.generator import Generator
from ssv2a.model.manifold import Manifold
from ssv2a.model.remixer import Remixer

# ── OpenAI VLM (GPT-4o) ──────────────────────────────────────────────────────
from ssv2a.model.openai_vlm import openai_classify_object

VLM_MODEL_NAME = "gpt-4o"


def gemini_filter_local_imgs(local_imgs: dict) -> tuple:
    """Filter local_imgs (4-tuples), removing silent sound sources."""
    print("\n🔎 OpenAI GPT-4o sound-source filtering …")
    total, kept, dropped = 0, 0, 0
    filter_log = []
    for img_key in list(local_imgs.keys()):
        filtered = []
        img_entry = {"image": os.path.basename(img_key), "kept": [], "dropped": []}
        for crop_path, locality, label, bbox in local_imgs[img_key]:
            total += 1
            crop_name = Path(crop_path).name
            decision = openai_classify_object(crop_path, img_key)
            if decision == "keep":
                filtered.append((crop_path, locality, label, bbox))
                kept += 1
                img_entry["kept"].append({"name": crop_name, "path": crop_path})
            else:
                dropped += 1
                img_entry["dropped"].append({"name": crop_name, "path": crop_path})
        local_imgs[img_key] = filtered
        filter_log.append(img_entry)

    print(f"\n   ✓ Kept {kept}/{total} objects, dropped {dropped} silent sound-sources.")
    print("   ┌─────────────────────────────────────────────")
    for entry in filter_log:
        print(f"   │ 📷 {entry['image']}")
        for k in entry["kept"]:
            print(f"   │    ✅ KEPT   : {k['name']}")
        for d in entry["dropped"]:
            print(f"   │    ❌ DROPPED: {d['name']}")
    print("   └─────────────────────────────────────────────\n")
    return local_imgs, filter_log


# ── SAM segmentation with bbox + center-point prompt ─────────────────────────

def sam_segment_kept_crops(local_imgs: dict, save_dir, sam_model: str = "sam_b.pt",
                           device: str = "cuda") -> dict:
    """Run SAM on each kept detection using YOLO bbox + center point as prompts.

    For each kept crop:
      1. Load the ORIGINAL image
      2. Compute center point: cx = (x1+x2)/2, cy = (y1+y2)/2
      3. Run SAM with bboxes + points=[cx,cy], labels=[1] (positive)
      4. Apply mask → save masked image
      5. Replace crop_path with SAM-masked image path
    """
    from ultralytics.models.sam import Predictor as SAMPredictor

    print("\n🎭 SAM segmentation on kept detections (bbox + center point) …")
    sam_save_dir = Path(save_dir) / 'sam_masks'
    os.makedirs(sam_save_dir, exist_ok=True)

    overrides = dict(
        conf=0.25, retina_masks=True, task="segment", mode="predict",
        model=sam_model, save=False, verbose=False, device=device,
    )
    sam = SAMPredictor(overrides=overrides)

    for img_key in local_imgs:
        sam.set_image(img_key)
        orig_img = np.array(Image.open(img_key))

        new_entries = []
        for crop_path, locality, label, bbox in local_imgs[img_key]:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            bbox_tensor = np.array([[x1, y1, x2, y2]])
            points = np.array([[cx, cy]])
            point_labels = np.array([1])

            results = sam(bboxes=bbox_tensor, points=points, labels=point_labels)

            if len(results) > 0 and results[0].masks is not None:
                mask = results[0].masks.data.cpu().numpy()
                mask = np.squeeze(mask, axis=0)
                if mask.ndim == 3:
                    mask = mask[0]
                mask = mask.astype(np.uint8)

                masked_img = orig_img * np.expand_dims(mask, axis=2)

                crop_name = Path(crop_path).stem
                sam_file = sam_save_dir / f'{crop_name}_sam.png'
                Image.fromarray(masked_img.astype(np.uint8)).save(sam_file, 'PNG')

                sam_crop = Image.open(crop_path).size
                sam_resized_file = sam_save_dir / f'{crop_name}_sam_resized.png'
                Image.fromarray(masked_img.astype(np.uint8)).crop(bbox).resize(
                    sam_crop, Image.Resampling.BICUBIC
                ).save(sam_resized_file, 'PNG')

                print(f"   🎭 {Path(crop_path).name}: SAM mask applied "
                      f"(bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] "
                      f"center=[{cx:.0f},{cy:.0f}]) → {sam_resized_file.name}")

                new_entries.append((str(sam_resized_file), locality, label, bbox))
            else:
                print(f"   ⚠️  {Path(crop_path).name}: SAM failed, using YOLO crop")
                new_entries.append((crop_path, locality, label, bbox))

        local_imgs[img_key] = new_entries
        sam.reset_image()

    del sam
    torch.cuda.empty_cache()
    return local_imgs


# ── BLIP captioning + YOLO label → sound description ─────────────────────────
_BLIP_CAP_CACHE = "/Data/taline.ouayjan/hf_cache/transformers/models--Salesforce--blip-image-captioning-base/snapshots/82a37760796d32b1411fe092ab5d4e227313294b"


def blip_caption_crops(local_imgs: dict, device: str = "cuda") -> dict:
    """Caption every kept crop (SAM-masked) with unconditional BLIP captioning.

    Args:
        local_imgs: { orig_path: [(sam_path, locality, label, bbox), ...] }
    Returns:
        captions: { orig_path: [caption_str, ...] }
    """
    from transformers import BlipProcessor, BlipForConditionalGeneration

    print("\n🖼  BLIP captioning SAM-masked crops (unconditional + YOLO label) …")
    processor = BlipProcessor.from_pretrained(_BLIP_CAP_CACHE, local_files_only=True)
    cap_model = BlipForConditionalGeneration.from_pretrained(
        _BLIP_CAP_CACHE, local_files_only=True,
    ).to(device)
    cap_model.eval()

    captions = {}
    for img_path in local_imgs:
        captions[img_path] = []
        for crop_path, _, label, _bbox in local_imgs[img_path]:
            img = Image.open(crop_path).convert("RGB")

            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = cap_model.generate(**inputs, max_new_tokens=30)
            raw_caption = processor.decode(out[0], skip_special_tokens=True).strip()

            caption = f"the sound of {raw_caption}"
            print(f"   📝 {Path(crop_path).name}: YOLO={label}  BLIP=\"{raw_caption}\" → \"{caption}\"")
            captions[img_path].append(caption)

    del cap_model, processor
    torch.cuda.empty_cache()
    return captions


# ── CLIP text embedding (768-dim, ViT-L/14) ─────────────────────────────────
def clip_embed_texts_l14(texts: list, batch_size: int = 256, device: str = "cuda"):
    with torch.no_grad():
        model, _ = clip_module.load("ViT-L/14", device=device)
        embeds = []
        for s in range(0, len(texts), batch_size):
            e = min(len(texts), s + batch_size)
            tokens = clip_module.tokenize(texts[s:e], truncate=True).to(device)
            text_features = model.encode_text(tokens)
            embeds.append(text_features)
        embeds = torch.cat(embeds).float()
        del model
        torch.cuda.empty_cache()
        return embeds / embeds.norm(p=2, dim=-1, keepdim=True)


# ── The Pipeline class ──────────────────────────────────────────────────────
from ssv2a.model.pipeline import Pipeline  # noqa: E402


# ── Multimodal image_to_audio: LLM filter + SAM + BLIP + text conditioning ──
@torch.no_grad()
def image_to_audio_mm2a_SAM(
    images, text="", transcription="", save_dir="", config=None,
    gen_remix=True, gen_tracks=False, emb_only=False,
    pretrained=None, batch_size=64, var_samples=1,
    shuffle_remix=True, cycle_its=3, cycle_samples=16, keep_data_cache=False,
    duration=10, seed=42, device='cuda',
):
    set_seed(seed)
    if not os.path.exists(config):
        config = Path().resolve() / 'configs' / 'model.json'
    with open(config, 'r') as fp:
        config = json.load(fp)

    if not save_dir:
        save_dir = Path().resolve() / 'output'
    else:
        save_dir = Path(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        if gen_tracks:
            os.makedirs(save_dir / 'tracks')
    cache_dir = save_dir / 'data_cache'

    # ── 1. YOLO detection (4-tuples with bbox) ──────────────────────────
    if not isinstance(images, dict):
        local_imgs = detect(
            images, config['detector'],
            save_dir=cache_dir / 'masked_images',
            batch_size=batch_size, device=device,
        )
    else:
        local_imgs = copy.deepcopy(images)
        images = [k for k in images]
        keep_data_cache = True

    # ── 2. ★ LLM FILTER — drop silent objects ───────────────────────────
    local_imgs, filter_log = gemini_filter_local_imgs(local_imgs)

    # Save filter report
    report_path = save_dir / 'filter_report.json'
    json_log = []
    for entry in filter_log:
        json_log.append({
            "image": entry["image"],
            "kept": [item["name"] for item in entry["kept"]],
            "dropped": [item["name"] for item in entry["dropped"]],
        })
    with open(report_path, 'w') as fp:
        json.dump({"model": VLM_MODEL_NAME, "results": json_log}, fp, indent=2)
    print(f"   📄 Filter report saved to {report_path}")

    # Copy crop images
    for entry in filter_log:
        img_name = Path(entry["image"]).stem
        img_kept_dir = save_dir / img_name / 'kept'
        img_dropped_dir = save_dir / img_name / 'dropped'
        os.makedirs(img_kept_dir, exist_ok=True)
        os.makedirs(img_dropped_dir, exist_ok=True)
        for item in entry["kept"]:
            if os.path.isfile(item["path"]):
                copy2(item["path"], img_kept_dir / item["name"])
        for item in entry["dropped"]:
            if os.path.isfile(item["path"]):
                copy2(item["path"], img_dropped_dir / item["name"])
        print(f"   📁 {img_name}/kept/    → {len(entry['kept'])} crops")
        print(f"   📁 {img_name}/dropped/ → {len(entry['dropped'])} crops")

    # If all crops dropped, skip
    images = [img for img in images if len(local_imgs.get(img, [])) > 0]
    if len(images) == 0:
        print("⚠  LLM filtered out ALL detected objects — nothing to generate.")
        if not keep_data_cache and cache_dir.exists():
            rmtree(cache_dir)
        return None

    # ── 3. ★ SAM SEGMENTATION — bbox + center point on kept crops ────────
    local_imgs = sam_segment_kept_crops(
        local_imgs, save_dir=cache_dir,
        sam_model=config['detector'].get('segment_model', 'sam_b.pt') or 'sam_b.pt',
        device=device,
    )

    # Save SAM-masked crops
    sam_kept_dir = save_dir / 'sam_kept'
    os.makedirs(sam_kept_dir, exist_ok=True)
    for img_key in local_imgs:
        for crop_path, _, label, bbox in local_imgs[img_key]:
            if os.path.isfile(crop_path):
                copy2(crop_path, sam_kept_dir / Path(crop_path).name)
    print(f"   📁 sam_kept/ → SAM-masked images saved")

    # ── 4. ★ BLIP captioning on SAM-masked crops ────────────────────────
    blip_captions = blip_caption_crops(local_imgs, device=device)

    # Save BLIP captions report
    captions_report = {}
    for img_path in blip_captions:
        img_name = Path(img_path).stem
        crop_names = [Path(cp).name for cp, _, _lbl, _bbox in local_imgs[img_path]]
        captions_report[img_name] = [
            {"crop": cn, "caption": cap}
            for cn, cap in zip(crop_names, blip_captions[img_path])
        ]
    captions_path = save_dir / 'blip_captions.json'
    with open(captions_path, 'w') as fp:
        json.dump(captions_report, fp, indent=2)
    print(f"   📄 BLIP captions saved to {captions_path}")

    # ── 5. CLIP embed SAM-masked images AND BLIP captions ────────────────
    global_clips = clip_embed_images(images, batch_size=batch_size, device=device)

    imgs = []
    all_captions_flat = []
    for img in images:
        for li, _, _lbl, _bbox in local_imgs[img]:
            imgs.append(li)
        all_captions_flat.extend(blip_captions[img])

    local_clips = clip_embed_images(imgs, batch_size=batch_size, device=device)
    text_clips = clip_embed_texts_l14(all_captions_flat, batch_size=batch_size, device=device)

    print(f"\n   🔗 Image CLIP shape: {local_clips.shape}, Text CLIP shape: {text_clips.shape}")

    # ── 6. Combine image + text CLIPs ────────────────────────────────────
    combined_clips_list = []
    combined_jumps = []
    offset = 0
    for img in images:
        n = len(local_imgs[img])
        for i in range(n):
            combined_clips_list.append(local_clips[offset + i])
            combined_clips_list.append(text_clips[offset + i])
        combined_jumps.append(n * 2)
        offset += n
    combined_clips = torch.stack(combined_clips_list)

    print(f"   🔗 Combined CLIP sources: {combined_clips.shape[0]} "
          f"({offset} image + {offset} text)")

    # ── 7. SSV2A manifold + remix ────────────────────────────────────────
    model = Pipeline(copy.deepcopy(config), pretrained, device)
    model.eval()
    with torch.no_grad():
        local_claps = model.clips2foldclaps(local_clips, var_samples=var_samples)

        if gen_remix:
            remix_clips = emb2seq(
                combined_jumps, combined_clips,
                max_length=model.remixer.slot, delay=1, device=model.device,
            )
            remix_clips[:, 0, :] = global_clips
            remix_clap = model.cycle_mix(
                remix_clips, its=cycle_its, var_samples=var_samples,
                samples=cycle_samples, shuffle=shuffle_remix,
            )
            del remix_clips

    if emb_only:
        if not keep_data_cache:
            rmtree(cache_dir)
        return remix_clap.detach().cpu().numpy()

    del local_clips, text_clips, combined_clips

    # ── 8. AudioLDM generation ───────────────────────────────────────────
    audioldm_v = config['audioldm_version']
    model = build_audioldm(model_name=audioldm_v, device=device)
    if gen_tracks:
        local_wave = emb_to_audio(model, local_claps, batchsize=batch_size, duration=duration)
    if gen_remix:
        waveform = emb_to_audio(model, remix_clap, batchsize=batch_size, duration=duration)

    # ── 9. I/O ───────────────────────────────────────────────────────────
    if gen_tracks:
        local_names = [Path(img).name.replace('.png', '') for img in imgs]
        save_wave(local_wave, save_dir / 'tracks', name=local_names)
    if gen_remix:
        save_wave(waveform, save_dir,
                  name=[os.path.basename(img).replace('.png', '') for img in images])
    if not keep_data_cache:
        rmtree(cache_dir)
