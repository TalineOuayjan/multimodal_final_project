"""
detect_spatial.py — YOLO detection that returns bounding-box coordinates for
spatial audio mixing.

Extends detect_gemini.py:
  - 4-tuples: (crop_path, locality, label, bbox)
  - bbox = [x1, y1, x2, y2] in absolute pixel coordinates
  - Also stores (img_width, img_height) as metadata on the returned dict
    so the spatial mixer knows the image dimensions.
"""

import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor
from PIL import Image
from tqdm.auto import tqdm

from ssv2a.data.utils import read_classes, mask2bbox


def yolo_detect(images, detection_model='yolov8x-worldv2.pt', segment_model="sam_b.pt",
                resize=None, crop=True, classes=None, batch_size=64, conf=.5, iou=0.5,
                max_det=64, top_k=None, save_dir="", device='cuda', **_):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = YOLO(detection_model)
    model.to(device)
    if 'world' in detection_model and classes is not None:
        classes = read_classes(classes)
        model.set_classes(classes)

    if resize is not None:
        imgsz = resize
    else:
        sample_img = Image.open(images[0])
        imgsz = sample_img.size
    img_area = imgsz[0] * imgsz[1]

    print(f"Detecting objects with {detection_model}:")
    segments = {}
    # Store image dimensions so the spatial mixer can normalise coordinates
    img_dimensions = {}
    for img in images:
        segments[img] = []

    for i in tqdm(range(0, len(images), batch_size)):
        e = min(len(images), i + batch_size)
        for img in images[i:e]:
            oimg = Image.open(img)
            if resize is not None and oimg.size != resize:
                oimg = oimg.resize(resize, resample=Image.Resampling.BICUBIC)
                oimg.save(img, 'PNG')
        detect_results = model.predict(images[i:e], imgsz=imgsz, conf=conf, iou=iou,
                                       max_det=max_det, augment=False,
                                       agnostic_nms=True, verbose=False)

        if crop:
            for j, img in enumerate(images[i:e]):
                oimg = Image.open(img)
                img_dimensions[img] = oimg.size          # (width, height)
                img_stem = Path(img).stem
                img_save_dir = Path(save_dir) / img_stem
                os.makedirs(img_save_dir, exist_ok=True)
                rs = detect_results[j][:top_k]
                # save annotated detection result
                annotated = detect_results[j].plot()
                Image.fromarray(annotated[..., ::-1]).save(
                    img_save_dir / f'{img_stem}_detections.png', 'PNG')
                for z, r in enumerate(rs):
                    box = r.boxes.xyxy.cpu().tolist()[0]      # [x1, y1, x2, y2]
                    cimg = oimg.crop(box).resize(imgsz, Image.Resampling.BICUBIC)
                    cimg_file = img_save_dir / f'{img_stem}_{z}.png'
                    cimg.save(cimg_file, 'PNG')
                    locality = abs(box[2] - box[0]) * abs(box[3] - box[1]) / img_area

                    cls_id = int(r.boxes.cls.cpu().tolist()[0])
                    label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                    segments[img].append((str(cimg_file), locality, label, box))

        else:
            overrides = dict(conf=.25, retina_masks=True, task="segment", mode="predict",
                             imgsz=imgsz, model=segment_model, save=False, verbose=False,
                             device=device)
            sam = SAMPredictor(overrides=overrides)
            for j in range(len(images[i:e])):
                sam.set_image(images[i:e][j])
                img_path = images[i:e][j]
                pil_img = Image.open(img_path)
                img_dimensions[img_path] = pil_img.size
                img_np = np.array(pil_img)
                img_stem = Path(img_path).stem
                img_save_dir = Path(save_dir) / img_stem
                os.makedirs(img_save_dir, exist_ok=True)
                annotated = detect_results[j].plot()
                Image.fromarray(annotated[..., ::-1]).save(
                    img_save_dir / f'{img_stem}_detections.png', 'PNG')
                rs = detect_results[j][:top_k]
                for z, r in enumerate(rs):
                    box = r.boxes.xyxy.cpu().tolist()[0]
                    mask = sam(bboxes=r.boxes.xyxy)[0].masks.data.cpu().numpy()
                    mask = np.squeeze(mask, axis=0).astype(int)
                    mimg_file = img_save_dir / f'{img_stem}_{z}.png'
                    Image.fromarray(
                        (img_np * np.expand_dims(mask, axis=2)).astype(np.uint8)
                    ).save(mimg_file, 'PNG')
                    locality = float(np.sum(mask.astype(int))) / img_area
                    cls_id = int(r.boxes.cls.cpu().tolist()[0])
                    label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                    segments[img_path].append((str(mimg_file), locality, label, box))
                sam.reset_image()

    # Attach image dimensions as an attribute (dict is mutable, can add attrs via subclass)
    segments = SpatialSegments(segments)
    segments.img_dimensions = img_dimensions
    return segments


class SpatialSegments(dict):
    """A dict subclass that carries extra ``img_dimensions`` metadata."""
    img_dimensions: dict = {}


def detect(images, detector_cfg, save_dir='masked_images', batch_size=64, device='cuda'):
    detector_cfg['save_dir'] = save_dir
    detector_cfg['batch_size'] = batch_size
    detector_cfg['device'] = device

    if 'yolo' in detector_cfg['detection_model']:
        return yolo_detect(images, **detector_cfg)
    else:
        raise NotImplementedError('Detection model is unsupported.')
