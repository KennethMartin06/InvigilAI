"""
phone_detector.py — Real-time phone detection using YOLOv8n (COCO).

COCO class 67 = "cell phone".  The model is loaded lazily on first call
and cached as a module-level singleton so it is not reloaded per frame.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

_PHONE_CLASS_ID = 67    # COCO index for "cell phone"
_MODEL_NAME     = "yolov8n.pt"
_CONF_THRESHOLD = 0.40  # minimum confidence to flag a detection

_yolo_model = None      # lazy singleton


def _load_model():
    """Load YOLOv8n (downloads ~6 MB on first run, then cached)."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO(_MODEL_NAME)
        logger.info("YOLOv8 phone detector loaded (COCO class 67 = cell phone)")
    except ImportError:
        logger.warning("ultralytics not installed — phone detection disabled")
        _yolo_model = None
    return _yolo_model


def detect_phone(image: np.ndarray) -> tuple[bool, float]:
    """
    Run YOLOv8 on an OpenCV BGR image and return phone detection result.

    Parameters
    ----------
    image : np.ndarray — BGR image (from cv2).

    Returns
    -------
    (detected, confidence)
        detected   : bool  — True if a phone is found above threshold.
        confidence : float — highest confidence score (0.0 if none).
    """
    model = _load_model()
    if model is None or image is None:
        return False, 0.0

    try:
        results = model(image, verbose=False, classes=[_PHONE_CLASS_ID])
        best_conf = 0.0
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
        detected = best_conf >= _CONF_THRESHOLD
        return detected, round(best_conf, 4)
    except Exception as exc:
        logger.warning("Phone detection error: %s", exc)
        return False, 0.0
