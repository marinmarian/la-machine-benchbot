"""Slot presence check with a fixed webcam, fully offline, no ML model.

Setup (camera fixed, all tools in their slots):
    python vision.py setup            # draw a box per slot, name them -> slots.json; that same
                                      # frame is saved as the "full" reference (so do it with tools in)
    python vision.py capture empty    # photo with every slot empty  (don't move the camera!)
    python vision.py check            # present / absent per slot (add a slot name for one)
    python vision.py check --show     # same, plus a window with the boxes drawn on the live frame
    python vision.py check --image f  # same, on a saved frame (no camera)
    python vision.py watch            # live window: every slot boxed + labelled, green/red, q to quit

How it works: each slot ROI is cropped from the live frame and from both reference
photos, downscaled to 32x32 grey and normalised (so room brightness changes cancel).
Whichever reference the live crop is closer to wins. It answers "is something in the
screwdriver slot", not "is this a screwdriver".

listen.py calls requirement_ok(skill) before each skill. `requires` in slots.json maps
skill -> {slot: "present" | "empty"}; a skill named like a slot defaults to present.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import detector

HERE = Path(__file__).parent
SLOTS_FILE = HERE / "slots.json"
REF_DIR = HERE / "vision"
REFS = {"full": REF_DIR / "full.jpg", "empty": REF_DIR / "empty.jpg"}
PATCH = (32, 32)


# ---------- config ----------

def load_config() -> dict | None:
    if not SLOTS_FILE.exists():
        return None
    return json.loads(SLOTS_FILE.read_text())


def available() -> bool:
    """True when a trained detector exists, or slots.json plus both reference photos do."""
    return detector.available() or (load_config() is not None and all(p.exists() for p in REFS.values()))


def backend() -> str:
    return "detector" if detector.available() else "slots"


def names() -> list[str]:
    """Things we can check for: detector classes, or slot names."""
    if detector.available():
        return list(detector._load().names.values())
    cfg = load_config()
    return list(cfg["slots"]) if cfg else []


# ---------- camera ----------

def grab_frame(cfg: dict, flush: int = 10) -> np.ndarray:
    """Open the camera, drop buffered frames, return one fresh frame, release."""
    cap = cv2.VideoCapture(cfg.get("camera_index", 0))
    if "width" in cfg:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {cfg.get('camera_index', 0)}")
    try:
        for _ in range(flush):
            cap.grab()
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError("camera returned no frame")
    if "width" in cfg and frame.shape[1::-1] != (cfg["width"], cfg["height"]):
        raise RuntimeError(f"camera gave {frame.shape[1]}x{frame.shape[0]}, slots.json expects "
                           f"{cfg['width']}x{cfg['height']}; re-run setup")
    return frame


class LiveCamera:
    """Background thread that keeps the newest frame. Use when a window stays open, so
    the check and the display share one camera handle instead of reopening it."""

    def __init__(self, cfg: dict):
        self.cfg, self.frame, self._stop = cfg, None, threading.Event()
        self.cap = cv2.VideoCapture(cfg.get("camera_index", 0))
        if "width" in cfg:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {cfg.get('camera_index', 0)}")
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if ok:
                self.frame = f
        self.cap.release()

    def latest(self, timeout: float = 3.0) -> np.ndarray:
        t = time.perf_counter()
        while self.frame is None and time.perf_counter() - t < timeout:
            time.sleep(0.02)
        if self.frame is None:
            raise RuntimeError("camera returned no frame")
        return self.frame

    def close(self):
        self._stop.set()


# ---------- the check ----------

def _signature(frame: np.ndarray, roi: list[int]) -> np.ndarray:
    x, y, w, h = roi
    crop = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    small = cv2.resize(crop, PATCH, interpolation=cv2.INTER_AREA).astype(np.float32)
    return (small - small.mean()) / (small.std() + 1e-6)


def check_frame(frame: np.ndarray, cfg: dict | None, refs: dict[str, np.ndarray] | None = None) -> dict[str, tuple[bool, float]]:
    """name -> (present, margin). Detector: margin = best confidence * 100 (>= 50 counts).
    Slots: margin > 0 means closer to the 'full' reference."""
    if detector.available():
        dets = detector.detect(frame, conf=0.05)
        out = {}
        for n in names():
            best = max((c for cls, c, _ in dets if cls == n), default=0.0)
            out[n] = (best >= detector.CONF, best * 100)
        return out
    refs = refs or {k: cv2.imread(str(p)) for k, p in REFS.items()}
    out = {}
    for slot, roi in cfg["slots"].items():
        live = _signature(frame, roi)
        d_full = float(np.linalg.norm(live - _signature(refs["full"], roi)))
        d_empty = float(np.linalg.norm(live - _signature(refs["empty"], roi)))
        out[slot] = (d_empty > d_full, d_empty - d_full)
    return out


def check(slot: str | None = None, image: str | None = None) -> dict[str, tuple[bool, float]]:
    cfg = load_config()
    if cfg is None and not detector.available():
        raise RuntimeError("no slots.json; run: python vision.py setup")
    cfg = cfg or {}
    frame = cv2.imread(image) if image else grab_frame(cfg)
    result = check_frame(frame, cfg)
    return {slot: result[slot]} if slot else result


def requirement(skill: str, cfg: dict | None) -> dict[str, str]:
    """name -> 'present'|'empty' that `skill` needs. Defaults to its namesake tool being present."""
    req = (cfg or {}).get("requires", {}).get(skill)
    if req is None and skill in names():
        req = {skill: "present"}
    return req or {}


def requirement_ok(skill: str, image: str | None = None, frame: np.ndarray | None = None) -> tuple[bool, str]:
    """(ok, reason). ok=True when nothing is required or everything required holds."""
    cfg = load_config()
    if not available():
        return True, ""
    req = requirement(skill, cfg)
    if not req:
        return True, ""
    if frame is None:
        frame = cv2.imread(image) if image else grab_frame(cfg or {})
    state = check_frame(frame, cfg)
    det = detector.available()
    for name, want in req.items():
        present, margin = state[name]
        nice = name.replace("_", " ")
        if want == "present" and not present:
            return False, (f"I don't see the {nice} on the bench" if det else f"there is nothing in the {nice} slot")
        if want == "empty" and present:
            return False, (f"the {nice} is still on the bench" if det else f"the {nice} slot is not empty")
    return True, ""


# ---------- CLI ----------

def cmd_setup(args):
    cfg = load_config() or {}
    cfg["camera_index"] = args.camera
    frame = grab_frame({"camera_index": args.camera})
    cfg["height"], cfg["width"] = frame.shape[:2]
    print("Draw a box around each tool slot (Enter after each, Esc when done).")
    rois = cv2.selectROIs("slots", frame, showCrosshair=False)
    cv2.destroyAllWindows()
    slots = {}
    for roi in rois:
        name = input(f"name for box {list(map(int, roi))} (e.g. screwdriver): ").strip().replace(" ", "_")
        if name:
            slots[name] = list(map(int, roi))
    cfg["slots"] = slots
    cfg.setdefault("requires", {})
    SLOTS_FILE.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"saved {len(slots)} slots -> {SLOTS_FILE}")
    REF_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(REFS["full"]), _with_boxes(frame, cfg))
    print(f"saved this frame as the 'full' reference -> {REFS['full']}  (open it: boxes must sit on the tools)")
    print("next: remove every tool, do NOT move the camera, then:  python vision.py capture empty")


def _with_boxes(frame: np.ndarray, cfg: dict | None, state: dict | None = None) -> np.ndarray:
    """Copy of frame with boxes and labels: detector boxes when a model exists, else slot boxes
    (green = present, red = absent, white = unknown)."""
    if detector.available():
        return detector.draw(frame)
    out = frame.copy()
    for slot, (x, y, w, h) in cfg["slots"].items():
        color, tag = (255, 255, 255), slot
        if state and slot in state:
            present, margin = state[slot]
            color = (0, 200, 0) if present else (0, 0, 255)
            tag = f"{slot} {margin:+.0f}"
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, tag, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def cmd_capture(args):
    cfg = load_config()
    if cfg is None:
        sys.exit("run setup first")
    REF_DIR.mkdir(exist_ok=True)
    frame = _with_boxes(grab_frame(cfg), cfg)
    cv2.imwrite(str(REFS[args.which]), frame)
    print(f"saved {REFS[args.which]}  ({frame.shape[1]}x{frame.shape[0]})")


def watch(cam: LiveCamera, cfg: dict, window: str = "benchbot camera (q to quit)", stop: threading.Event | None = None) -> None:
    """Show the live frame with every slot boxed and labelled until q is pressed or `stop` is set."""
    refs = None if detector.available() else {k: cv2.imread(str(p)) for k, p in REFS.items()}
    while not (stop and stop.is_set()):
        frame = cam.latest()
        state = check_frame(frame, cfg, refs)
        cv2.imshow(window, _with_boxes(frame, cfg, state))
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def cmd_watch(args):
    cfg = load_config()
    if not available():
        sys.exit("no detector (collect.py + train.py) and no slots (setup + capture empty)")
    print(f"backend: {backend()}  classes: {names()}")
    cam = LiveCamera(cfg or {"camera_index": 0})
    try:
        watch(cam, cfg)
    finally:
        cam.close()


def cmd_check(args):
    cfg = load_config()
    if not available():
        sys.exit("no detector (collect.py + train.py) and no slots (setup + capture empty)")
    print(f"backend: {backend()}")
    t = time.perf_counter()
    frame = cv2.imread(args.image) if args.image else grab_frame(cfg or {})
    state = check_frame(frame, cfg)
    if args.slot:
        state = {args.slot: state[args.slot]}
    for slot, (present, margin) in state.items():
        print(f"  {slot:14s} {'PRESENT' if present else 'absent ':8s} margin {margin:+.1f}")
    print(f"  ({time.perf_counter() - t:.2f}s)" + ("" if detector.available() else "   |margin| < 10 is unreliable: tighten that box or re-capture"))
    if args.show:
        cv2.imshow("slots (any key to close)", _with_boxes(frame, cfg, state))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup"); s.add_argument("--camera", type=int, default=0); s.set_defaults(fn=cmd_setup)
    c = sub.add_parser("capture"); c.add_argument("which", choices=list(REFS)); c.set_defaults(fn=cmd_capture)
    w = sub.add_parser("watch"); w.set_defaults(fn=cmd_watch)
    k = sub.add_parser("check"); k.add_argument("slot", nargs="?"); k.add_argument("--image"); k.add_argument("--show", action="store_true"); k.set_defaults(fn=cmd_check)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
