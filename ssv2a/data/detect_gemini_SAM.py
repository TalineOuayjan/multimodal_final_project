"""
detect_gemini_SAM.py — YOLO detection returning 4-tuples with bounding boxes.

Identical to detect_gemini.py except:
  - yolo_detect() stores 4-tuples (crop_path, locality, label, bbox)
  - bbox is the raw YOLO bounding box [x1, y1, x2, y2] needed for SAM prompting
"""

import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from PIL import Image
from tqdm.auto import tqdm

from ssv2a.data.utils import read_classes


def yolo_detect(images, detection_model='yolov8x-worldv2.pt', segment_model="sam_b.pt", resize=None, crop=True,
                classes=None, batch_size=64, conf=.5, iou=0.5, max_det=64, top_k=None, save_dir="", device='cuda', **_):
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
    for img in images:
        segments[img] = []

    for i in tqdm(range(0, len(images), batch_size)):
        e = min(len(images), i + batch_size)
        for img in images[i:e]:
            oimg = Image.open(img)
            if resize is not None and oimg.size != resize:
                oimg = oimg.resize(resize, resample=Image.Resampling.BICUBIC)
                oimg.save(img, 'PNG')
        detect_results = model.predict(images[i:e], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                                       augment=True, verbose=False)

        for j, img in enumerate(images[i:e]):
            oimg = Image.open(img)
            img_stem = Path(img).stem
            img_save_dir = Path(save_dir) / img_stem
            os.makedirs(img_save_dir, exist_ok=True)
            rs = detect_results[j][:top_k]
            # save annotated detection result
            annotated = detect_results[j].plot()
            Image.fromarray(annotated[..., ::-1]).save(img_save_dir / f'{img_stem}_detections.png', 'PNG')
            for z, r in enumerate(rs):
                box = r.boxes.xyxy.cpu().tolist()[0]  # [x1, y1, x2, y2]
                cimg = oimg.crop(box).resize(imgsz, Image.Resampling.BICUBIC)
                cimg_file = img_save_dir / f'{img_stem}_{z}.png'
                cimg.save(cimg_file, 'PNG')
                locality = abs(box[2] - box[0]) * abs(box[3] - box[1]) / img_area

                cls_id = int(r.boxes.cls.cpu().tolist()[0])
                label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                # Store 4-tuple: (crop_path, locality, label, bbox)
                segments[img].append((str(cimg_file), locality, label, box))

    return segments


def detect(images, detector_cfg, save_dir='masked_images', batch_size=64, device='cuda'):
    detector_cfg['save_dir'] = save_dir
    detector_cfg['batch_size'] = batch_size
    detector_cfg['device'] = device

    if 'yolo' in detector_cfg['detection_model']:
        return yolo_detect(images, **detector_cfg)
    else:
        raise NotImplementedError('Detection model is unsupported.')
