"""Fine-tune a small YOLO on the frames collected with collect.py.

    python train.py                  # ~40 epochs on the Mac GPU (mps), writes weights/handy.pt
    python train.py --epochs 60 --device cpu

Needs weights/yolov8n.pt (pretrained start, 6 MB) and dataset/ from collect.py. Fully offline.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

HERE = Path(__file__).parent
DS = HERE / "dataset"
OUT = HERE / "weights" / "handy.pt"
BASE = HERE / "weights" / "yolov8n.pt"


def build_split(val_frac: float = 0.15, seed: int = 0) -> Path:
    """Write dataset/data.yaml with a random train/val split (as txt lists; no files copied)."""
    classes = (DS / "classes.txt").read_text().split()
    images = sorted((DS / "images").glob("*.jpg"))
    if not images:
        raise SystemExit("dataset/images is empty; run collect.py first")
    random.Random(seed).shuffle(images)
    n_val = max(1, int(len(images) * val_frac))
    (DS / "val.txt").write_text("\n".join(str(p.resolve()) for p in images[:n_val]) + "\n")
    (DS / "train.txt").write_text("\n".join(str(p.resolve()) for p in images[n_val:]) + "\n")
    yaml = DS / "data.yaml"
    yaml.write_text(
        f"path: {DS.resolve()}\ntrain: train.txt\nval: val.txt\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes))
    )
    per_class = {c: len(list((DS / "images").glob(f"{c}_*.jpg"))) for c in classes}
    bg = len(list((DS / "images").glob("bg_*.jpg")))
    print(f"{len(images)} frames ({n_val} val). per class: {per_class}, background: {bg}")
    return yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps", help="mps | cpu")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from ultralytics import YOLO

    yaml = build_split()
    model = YOLO(str(BASE))
    model.train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, device=args.device, batch=args.batch,
                plots=False, workers=0, project=str(HERE / "runs"), name="handy", exist_ok=True)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    shutil.copy(best, OUT)
    print(f"\nsaved detector -> {OUT}")
    print("check it:  python vision.py watch      (boxes now come from the detector)")


if __name__ == "__main__":
    main()
