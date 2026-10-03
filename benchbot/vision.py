"""Slot presence check with a fixed webcam, fully offline, no ML model.

Setup (camera fixed, all tools in their slots):
    python vision.py setup            # draw a box per slot, name them -> slots.json
    python vision.py capture full     # photo with every tool in its slot
    python vision.py capture empty    # photo with every slot empty
    python vision.py check            # present / absent per slot (add a slot name for one)
    python vision.py check --image f  # same, on a saved frame (no camera)

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
import time
from pathlib import Path

import cv2
import numpy as np

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
    """True when slots.json and both reference photos exist."""
    return load_config() is not None and all(p.exists() for p in REFS.values())


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


# ---------- the check ----------

def _signature(frame: np.ndarray, roi: list[int]) -> np.ndarray:
    x, y, w, h = roi
    crop = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    small = cv2.resize(crop, PATCH, interpolation=cv2.INTER_AREA).astype(np.float32)
    return (small - small.mean()) / (small.std() + 1e-6)


def check_frame(frame: np.ndarray, cfg: dict, refs: dict[str, np.ndarray] | None = None) -> dict[str, tuple[bool, float]]:
    """slot -> (present, margin). margin > 0 means closer to the 'full' reference."""
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
    if cfg is None:
        raise RuntimeError("no slots.json; run: python vision.py setup")
    frame = cv2.imread(image) if image else grab_frame(cfg)
    result = check_frame(frame, cfg)
    return {slot: result[slot]} if slot else result


def requirement(skill: str, cfg: dict) -> dict[str, str]:
    """slot -> 'present'|'empty' that `skill` needs. Defaults to its namesake slot being present."""
    req = cfg.get("requires", {}).get(skill)
    if req is None and skill in cfg["slots"]:
        req = {skill: "present"}
    return req or {}


def requirement_ok(skill: str, image: str | None = None) -> tuple[bool, str]:
    """(ok, reason). ok=True when nothing is required or everything required holds."""
    cfg = load_config()
    if cfg is None:
        return True, ""
    req = requirement(skill, cfg)
    if not req:
        return True, ""
    frame = cv2.imread(image) if image else grab_frame(cfg)
    state = check_frame(frame, cfg)
    for slot, want in req.items():
        present, margin = state[slot]
        if want == "present" and not present:
            return False, f"there is nothing in the {slot.replace('_', ' ')} slot"
        if want == "empty" and present:
            return False, f"the {slot.replace('_', ' ')} slot is not empty"
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
    print("next: python vision.py capture full   (then remove the tools)   python vision.py capture empty")


def cmd_capture(args):
    cfg = load_config()
    if cfg is None:
        sys.exit("run setup first")
    REF_DIR.mkdir(exist_ok=True)
    frame = grab_frame(cfg)
    for slot, (x, y, w, h) in cfg["slots"].items():
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imwrite(str(REFS[args.which]), frame)
    print(f"saved {REFS[args.which]}  ({frame.shape[1]}x{frame.shape[0]})")


def cmd_check(args):
    t = time.perf_counter()
    for slot, (present, margin) in check(args.slot, args.image).items():
        print(f"  {slot:14s} {'PRESENT' if present else 'absent ':8s} margin {margin:+.1f}")
    print(f"  ({time.perf_counter() - t:.2f}s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup"); s.add_argument("--camera", type=int, default=0); s.set_defaults(fn=cmd_setup)
    c = sub.add_parser("capture"); c.add_argument("which", choices=list(REFS)); c.set_defaults(fn=cmd_capture)
    k = sub.add_parser("check"); k.add_argument("slot", nargs="?"); k.add_argument("--image"); k.set_defaults(fn=cmd_check)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
