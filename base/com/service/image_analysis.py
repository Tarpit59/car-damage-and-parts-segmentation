"""
image_analysis.py
=================
YOLO-based car bounding-box detector.
"""

import os
import cv2
from ultralytics import YOLO

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import YOLO_CHECKPOINT, PADDING_RATIO as DEFAULT_PADDING

# Load once at import time
yolo_model = YOLO(YOLO_CHECKPOINT)


def analyze_single_car_image(image_path, padding_ratio=DEFAULT_PADDING):
    """
    Detect the biggest car in the image and return expanded ROI coordinates.

    Returns
    -------
    success : bool
    result  : dict  {image, coords}
    """

    def get_main_car(cars_detected):
        return max(
            cars_detected,
            key=lambda box: (
                (box.xyxy[0][2] - box.xyxy[0][0]) *
                (box.xyxy[0][3] - box.xyxy[0][1])
            )
        )

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return False, {}

    image  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    results    = yolo_model(image, verbose=False)
    detections = results[0].boxes

    # COCO class 2 = car
    cars_detected = [box for box in detections if int(box.cls.item()) == 2]
    if not cars_detected:
        return False, {}

    main_car     = get_main_car(cars_detected)
    x1, y1, x2, y2 = main_car.xyxy[0].cpu().numpy().astype(int)

    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)

    return True, {
        "image":  os.path.basename(image_path),
        "coords": [x1, y1, x2, y2],
    }
