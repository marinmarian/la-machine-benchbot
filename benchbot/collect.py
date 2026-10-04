"""Collect auto-labelled training frames for the tool detector, one object class at a time.

    python collect.py --background            # 1) empty bench (camera fixed!): (re)writes vision/empty.jpg
                                              #    + ~20 empty frames.
    python collect.py --background --keep-empty   # then arms parked IN view, bench still empty: more
                                              #    background frames so the detector learns arms are not tools
    python collect.py screwdriver --seconds 45   # 2) ONE object on the bench, move it around while it records
    python collect.py microphone                 #    repeat per class (class name = skill name)

Labels come from background subtraction against vision/empty.jpg: the single largest blob
is the object. Frames with no blob, two blobs, or a blob touching the border are skipped.
Output: dataset/images/*.jpg + dataset/labels/*.txt (YOLO format) + dataset/classes.txt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import vision

HERE = Path(__file__).parent
DS = HERE / "dataset"
CLASSES_FILE = DS / "classes.txt"
EMPTY = vision.REFS["empty"]


def classes() -> list[str]:
    return CLASSES_FILE.read_text().split() if CLASSES_FILE.exists() else []


def class_id(name: str) -> int:
    names = classes()
    if name not in names:
        names.append(name)
        CLASSES_FILE.write_text("\n".join(names) + "\n")
    return names.index(name)


def find_object(frame: np.ndarray, background: np.ndarray, min_area: int = 1500, max_frac: float = 0.25):
    """Return (x, y, w, h) of the single foreground object, or None if ambiguous."""
    def prep(im):
        return cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    th = cv2.threshold(cv2.absdiff(prep(frame), prep(background)), 25, 255, cv2.THRESH_BINARY)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    H, W = th.shape
    _, _, stats, _ = cv2.connectedComponentsWithStats(th)
    blobs = [s for s in stats[1:] if s[4] >= min_area]
    blobs = [s for s in blobs if s[0] > 3 and s[1] > 3 and s[0] + s[2] < W - 3 and s[1] + s[3] < H - 3]  # not on the border
    blobs = [s for s in blobs if s[2] * s[3] < max_frac * H * W]
    if len(blobs) != 1:
        return None
    x, y, w, h, _ = blobs[0]
    return int(x), int(y), int(w), int(h)


def save(frame: np.ndarray, label: str, stem: str) -> None:
    (DS / "images").mkdir(parents=True, exist_ok=True)
    (DS / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DS / "images" / f"{stem}.jpg"), frame)
    (DS / "labels" / f"{stem}.txt").write_text(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", help="class name (= skill name), e.g. screwdriver")
    ap.add_argument("--background", action="store_true", help="record empty-bench / arm-only frames with empty labels")
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--every", type=float, default=0.3, help="seconds between saved frames")
    ap.add_argument("--keep-empty", action="store_true", help="--background: keep the existing vision/empty.jpg (use for the arm-in-view pass)")
    args = ap.parse_args()
    if not args.background and not args.name:
        ap.error("give a class name or --background")

    cfg = vision.load_config() or {"camera_index": 0}
    cam = vision.LiveCamera(cfg)
    stem_base = f"{int(time.time())}"
    try:
        if args.background:
            frame = cam.latest()
            if not args.keep_empty:
                # always refresh: a stale empty.jpg from an earlier camera position poisons every label
                EMPTY.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(EMPTY), frame)
                print(f"saved background -> {EMPTY}  (bench must be EMPTY for this one)")
            print("recording background frames (empty bench, or arm in view, nothing else)…")
            n, t0, last = 0, time.perf_counter(), 0.0
            while time.perf_counter() - t0 < args.seconds / 3:
                frame = cam.latest()
                cv2.imshow("collect (q to stop)", frame)
                if time.perf_counter() - last >= args.every:
                    save(frame, "", f"bg_{stem_base}_{n:04d}"); n += 1; last = time.perf_counter()
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            print(f"saved {n} background frames")
            return

        if not EMPTY.exists():
            raise SystemExit(f"no {EMPTY}; run: python collect.py --background  (empty bench first)")
        background = cv2.imread(str(EMPTY))
        cid = class_id(args.name)
        print(f"class '{args.name}' id {cid}. Put ONLY the {args.name} on the bench and keep moving/rotating it.")
        n, skipped, t0, last = 0, 0, time.perf_counter(), 0.0
        while time.perf_counter() - t0 < args.seconds:
            frame = cam.latest()
            box = find_object(frame, background)
            show = frame.copy()
            if box:
                x, y, w, h = box
                cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(show, f"{args.name}  saved {n}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(show, "no single object found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("collect (q to stop)", show)
            if time.perf_counter() - last >= args.every:
                last = time.perf_counter()
                if box:
                    H, W = frame.shape[:2]
                    x, y, w, h = box
                    save(frame, f"{cid} {(x + w / 2) / W:.5f} {(y + h / 2) / H:.5f} {w / W:.5f} {h / H:.5f}\n",
                         f"{args.name}_{stem_base}_{n:04d}")
                    n += 1
                else:
                    skipped += 1
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        print(f"saved {n} frames for '{args.name}', skipped {skipped} ambiguous ones -> {DS}")
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
