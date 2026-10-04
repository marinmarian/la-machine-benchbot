"""YOLO tool detector trained with train.py (weights/handy.pt). Used by vision.py when present.

    python detector.py vision/bench_test_frame.jpg     # print detections on a saved frame
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = Path(__file__).parent / "weights" / "handy.pt"
CONF = 0.5          # minimum confidence for a detection to count as "the tool is on the bench"
_model = None


def available() -> bool:
    return WEIGHTS.exists()


def _load():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(str(WEIGHTS))
    return _model


def detect(frame: np.ndarray, conf: float = CONF) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """[(class_name, confidence, (x1, y1, x2, y2)), ...] above `conf`."""
    m = _load()
    r = m.predict(frame, conf=conf, imgsz=640, device="cpu", verbose=False)[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
        out.append((m.names[int(b.cls)], float(b.conf), (x1, y1, x2, y2)))
    return out


def present(frame: np.ndarray, name: str, conf: float = CONF) -> tuple[bool, float]:
    """(seen, best_confidence) for class `name` in the frame."""
    scores = [c for cls, c, _ in detect(frame, conf=0.05) if cls == name]
    best = max(scores, default=0.0)
    return best >= conf, best


def draw(frame: np.ndarray, dets=None) -> np.ndarray:
    """Copy of frame with a labelled box per detection."""
    out = frame.copy()
    for name, c, (x1, y1, x2, y2) in (dets if dets is not None else detect(frame)):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f"{name} {c:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    return out


if __name__ == "__main__":
    if not available():
        sys.exit(f"no {WEIGHTS}; run collect.py then train.py")
    img = cv2.imread(sys.argv[1])
    for name, c, box in detect(img):
        print(f"  {name:14s} {c:.2f}  {box}")
