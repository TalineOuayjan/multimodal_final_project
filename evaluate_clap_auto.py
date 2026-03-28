"""
evaluate_clap_auto.py — Fully automated CLAP evaluation.

Reads the filter_report.json + kept/dropped crop images from the filtered
pipeline output, sends them to an OpenAI GPT-4o VLM to
auto-generate positive and negative text descriptions, then runs CLAP
alignment scoring.

Compare the original audio against as many other audios as you want in one command:

  python evaluate_clap_auto.py \\
      --gemini_dir output_gemini/DOG-LION \\
      --image_name DOG-LION \\
      --original_audio output/DOG-LION/DOG-LION.wav \\
      --compare_audio output_gemini/DOG-LION/DOG-LION.wav output_mm2a_gemini_BLIP/DOG-LION/DOG-LION.wav \\
      --labels "Original" "Gemini" "MM2A-BLIP"

  --labels is optional; if omitted, files are named Audio-1, Audio-2, …
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from ssv2a.model.clap import CLAP
from ssv2a.data.utils import normalize_wav
from ssv2a.model.openai_vlm import openai_classify_object, FILTER_PROMPT


# ── Prompt generation ─────────────────────────────────────────────────────────

def describe_crop(image_path: str) -> str:
    """Ask GPT-4o to describe the sound an object in a crop would make."""
    import os, base64, openai
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = Path(image_path).suffix.lower().replace(".", "") or "jpeg"
    mime = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "webp") else "image/jpeg"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "You are an expert in audio-visual scene analysis. "
                "Given a cropped image of an object, describe in 3–6 words "
                "the specific sound this object produces (e.g. 'dog barking', 'lion roaring', 'car engine running'). "
                "If silent, write 'silence'. Answer with ONLY the short description."
            )},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}},
                {"type": "text", "text": "What sound does this object make?"},
            ]},
        ],
        max_tokens=20,
        temperature=0.0,
        seed=42,
    )
    return response.choices[0].message.content.strip().lower()


def generate_prompts(kept_dir: str, dropped_dir: str) -> tuple:
    """Generate positive/negative prompts from kept/dropped crop images."""
    kept_imgs = sorted(Path(kept_dir).glob("*.*"))
    dropped_imgs = sorted(Path(dropped_dir).glob("*.*"))

    print()
    kept_descs = [describe_crop(str(p)) for p in kept_imgs if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    kept_descs = [d for d in kept_descs if d and d != "silence"]
    print(f"  🔊 Kept sounds:    {kept_descs}")

    dropped_descs = [describe_crop(str(p)) for p in dropped_imgs if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    dropped_descs = [d for d in dropped_descs if d and d != "silence"]
    print(f"  🔇 Dropped sounds: {dropped_descs}")

    positive = ", ".join(kept_descs) if kept_descs else "ambient background sounds"
    negative = ", ".join(dropped_descs) if dropped_descs else "complete silence"

    print(f"\n  📝 Positive prompt: {positive}")
    print(f"  📝 Negative prompt: {negative}\n")
    return positive, negative


# ── CLAP helpers ──────────────────────────────────────────────────────────────

def load_audio_embedding(clap_model, path, device):
    waveform, sr = torchaudio.load(path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    waveform = waveform.numpy()[0, ...]
    waveform = normalize_wav(waveform)
    waveform = waveform[None, ...]
    waveform = waveform / np.max(np.abs(waveform))
    waveform = np.nan_to_num(0.5 * waveform)
    emb = clap_model.model(torch.from_numpy(waveform).float().to(device))
    return emb.detach().cpu().squeeze()


def load_text_embedding(clap_model, text):
    emb = clap_model.model([text])
    return emb.detach().cpu().squeeze()


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Automated CLAP evaluation: compare original audio vs N other audios."
    )
    parser.add_argument("--gemini_dir", required=True,
                        help="Path to the filtered output dir (contains <image_name>/kept/ and <image_name>/dropped/)")
    parser.add_argument("--image_name", required=True,
                        help="Name of the image (without extension), e.g. DOG-LION")
    parser.add_argument("--original_audio", required=True,
                        help="Path to the original/reference audio file")
    parser.add_argument("--compare_audio", nargs="+", default=[],
                        help="One or more audio files to compare against the original")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Optional labels for all audio files (original first, then compare). "
                             "If omitted, files are named 'Original', 'Audio-1', 'Audio-2', …")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    gemini_dir = Path(args.gemini_dir)

    # Build list of all audio files and their labels
    all_audio = [args.original_audio] + args.compare_audio
    if args.labels:
        if len(args.labels) != len(all_audio):
            print(f"WARNING: {len(args.labels)} labels given for {len(all_audio)} audio files — ignoring labels.")
            all_labels = ["Original"] + [f"Audio-{i+1}" for i in range(len(args.compare_audio))]
        else:
            all_labels = args.labels
    else:
        all_labels = ["Original"] + [f"Audio-{i+1}" for i in range(len(args.compare_audio))]

    # Validate directories
    kept_dir = gemini_dir / args.image_name / "kept"
    dropped_dir = gemini_dir / args.image_name / "dropped"
    if not kept_dir.exists():
        print(f"ERROR: {kept_dir} not found"); return
    if not dropped_dir.exists():
        print(f"ERROR: {dropped_dir} not found"); return

    # ── Step 1: Auto-generate prompts ────────────────────────────────────
    print("Step 1: Generating CLAP prompts from crop images via OpenAI VLM …")
    positive, negative = generate_prompts(str(kept_dir), str(dropped_dir))

    # ── Step 2: Load CLAP ────────────────────────────────────────────────
    print("Step 2: Loading CLAP models …")
    clap_text  = CLAP(clap_version="audioldm-s-full-v2", embed_mode="text",  device=device)
    clap_audio = CLAP(clap_version="audioldm-s-full-v2", embed_mode="audio", device=device)

    pos_emb = load_text_embedding(clap_text, positive)
    neg_emb = load_text_embedding(clap_text, negative)

    # ── Step 3: Evaluate all audio files ─────────────────────────────────
    print(f"Step 3: Evaluating {len(all_audio)} audio files …")
    results = []
    for audio_path, label in zip(all_audio, all_labels):
        if not os.path.isfile(audio_path):
            print(f"  ⚠  Skipping {label}: file not found ({audio_path})")
            continue
        audio_emb   = load_audio_embedding(clap_audio, audio_path, device)
        pos_score   = cosine_sim(audio_emb, pos_emb)
        neg_score   = cosine_sim(audio_emb, neg_emb)
        neg_clamped = max(neg_score, 0.0)
        alignment   = pos_score - neg_clamped
        results.append({
            "label":       label,
            "path":        audio_path,
            "positive":    pos_score,
            "negative":    neg_score,
            "neg_clamped": neg_clamped,
            "alignment":   alignment,
        })

    # ── Print results table ───────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"  Positive (should hear)     : {positive}")
    print(f"  Negative (should NOT hear) : {negative}")
    print(f"  [auto-generated by OpenAI GPT-4o from kept/dropped crops]")
    print("=" * 100)

    col_label = max(len(r["label"]) for r in results) + 2
    col_path  = min(max(len(r["path"])  for r in results) + 2, 50)
    header = f"{'Pipeline':<{col_label}} {'Audio File':<{col_path}} {'Pos ↑':>8} {'Neg ↓':>8} {'Align ↑':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        path_display = r["path"] if len(r["path"]) <= col_path else "…" + r["path"][-(col_path-1):]
        print(f"{r['label']:<{col_label}} {path_display:<{col_path}} {r['positive']:>8.4f} {r['negative']:>8.4f} {r['alignment']:>9.4f}")

    print()
    best = max(results, key=lambda x: x["alignment"])
    print(f"  🏆 Best alignment: {best['label']}")
    print(f"     Positive = {best['positive']:.4f}  |  Negative = {best['negative']:.4f}  |  Alignment = {best['alignment']:.4f}")

    # ── Save results ─────────────────────────────────────────────────────
    report = {
        "positive_prompt": positive,
        "negative_prompt": negative,
        "results": results,
    }
    out_path = gemini_dir / "clap_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  📄 Results saved to {out_path}")


if __name__ == "__main__":
    main()
