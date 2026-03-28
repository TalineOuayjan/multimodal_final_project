import os
import base64
import openai
from dotenv import load_dotenv
load_dotenv()

# Centralized prompt for object classification
FILTER_PROMPT = (
    "I'm building a sound-generation pipeline from images. I will show you two images:\n"
    "1. The FULL original scene image (for context).\n"
    "2. A CROPPED object detected from that scene.\n\n"
    "Your job: decide whether the cropped object should be KEPT or DROPPED from "
    "the audio generation pipeline.\n\n"
    "Rules:\n"
    "- Objects that are inherently sound-producing AND appear to be ACTIVE → 'keep' "
    "(example: a person talking with open mouth, a running engine, a barking dog, a musical instrument being played). "
    "These are actively making sound — keep them.\n"
    "- Objects that are inherently sound-producing BUT appear INACTIVE/SILENT in this image → 'drop' "
    "(example: a person with closed mouth standing still, a parked car with engine off, a silent TV). "
    "These could make sound but aren't — drop them.\n"
    "- Objects that are NOT inherently sound-producing → 'keep' "
    "(example: jeans, jacket, shoes, chair, wall, bag, hat, table, food, plants). "
    "These are background/context objects — always keep them.\n\n"
    "Answer with ONLY one word: 'keep' or 'drop'."
)


def _encode_image(image_path: str) -> str:
    """Base64-encode an image file for the OpenAI Vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def openai_classify_object(crop_path, context=None, api_key=None, model="gpt-4o"):
    """
    Use OpenAI GPT-4o Vision to classify if a cropped object is producing sound.
    Sends the actual crop image (and optionally the full scene image) to the model.

    Args:
        crop_path (str): Path to the cropped object image.
        context (str, optional): Path to the full scene image for context.
        api_key (str, optional): OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model (str): OpenAI model to use (default: gpt-4o).
    Returns:
        str: 'keep' or 'drop'
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in your environment or pass api_key.")

    client = openai.OpenAI(api_key=api_key)

    # Build the vision message content
    content = []

    # Add full scene image if available
    if context is not None and os.path.isfile(context):
        scene_b64 = _encode_image(context)
        ext = os.path.splitext(context)[-1].lower().replace(".", "") or "jpeg"
        mime = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "webp", "gif") else "image/jpeg"
        content.append({"type": "text", "text": "Image 1 — Full scene (for context):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{scene_b64}", "detail": "low"}})

    # Add cropped object image
    if os.path.isfile(crop_path):
        crop_b64 = _encode_image(crop_path)
        ext = os.path.splitext(crop_path)[-1].lower().replace(".", "") or "jpeg"
        mime = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "webp", "gif") else "image/jpeg"
        content.append({"type": "text", "text": "Image 2 — Cropped object (decide keep or drop):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{crop_b64}", "detail": "high"}})
    else:
        # Fallback: if file not found, use filename as text hint
        content.append({"type": "text", "text": f"Cropped object filename: {os.path.basename(crop_path)}"})

    content.append({"type": "text", "text": "Answer with ONLY one word: 'keep' or 'drop'."})

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": FILTER_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=5,
        temperature=0.0,
        seed=42,
        n=1,
    )

    answer = response.choices[0].message.content.strip().lower()
    if "keep" in answer:
        return "keep"
    elif "drop" in answer:
        return "drop"
    else:
        # Default to keep if ambiguous
        return "keep"
