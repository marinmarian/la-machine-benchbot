"""Shift part of a recorded skill in space, e.g. grip a few millimetres lower.

    python adjust.py screwdriver4 --dz -0.3 --window 164 180 250 282 --dry-run   # report only
    python adjust.py screwdriver4 --dz -0.3 --window 164 180 250 282

--window a b c e (frame indices): unchanged before a, eases in over a..b, full shift from b to c,
eases out over c..e, unchanged after e. --dx/--dy/--dz are centimetres in the robot base frame
(z up). Each frame in the window: forward kinematics -> add w * d -> inverse kinematics seeded from
the previous frame, so neighbouring frames stay close. wrist_roll and gripper are copied from the
recording (alignment is position-only, see README).

Kinematics: TheRobotStudio's so101_new_calib.urdf (visual/collision meshes stripped), whose zero
is lerobot's calibrated middle-of-range, so skill degrees go in directly.
The original skill is backed up to skills/backup/ before it is overwritten; every adjustment is
also logged in the skill's "adjustments" list.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

import arm

URDF = Path(__file__).parent / "urdf" / "so101_new_calib_kinematics.urdf"
SOLVE = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
MAX_RESIDUAL_M = 0.003   # four joints trade 1-2 mm against keeping the gripper vertical; a grasp tolerates that


class Kinematics:
    """FK/IK over the four solved joints, wrist_roll held at a given value."""

    def __init__(self, orientation_weight: float = 0.1):
        from lerobot.model.kinematics import RobotKinematics
        self.k = RobotKinematics(str(URDF), "gripper_frame_link", SOLVE)
        for j in ("wrist_roll", "gripper"):
            self.k.solver.mask_dof(j)
        self.orientation_weight = orientation_weight
        self.pan_xy = self.k.robot.get_T_world_frame("shoulder_link")[:2, 3].copy()   # pan axis: vertical, through here

    def target(self, T0: np.ndarray, d: np.ndarray) -> np.ndarray:
        """T0 moved by d (metres), turned about the pan axis by the yaw a move to there forces.

        Four joints can't move the gripper sideways without pan turning it too, so asking for T0's
        exact orientation would leave IK stuck halfway (wrist_roll is not used, see README).
        """
        T = T0.copy()
        T[:3, 3] += d
        a, b = T0[:2, 3] - self.pan_xy, T[:2, 3] - self.pan_xy
        yaw = np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0])
        c, s = np.cos(yaw), np.sin(yaw)
        T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ T0[:3, :3]
        return T

    def fk(self, q: np.ndarray, roll: float) -> np.ndarray:
        self.k.robot.set_joint("wrist_roll", np.deg2rad(roll))
        return self.k.forward_kinematics(q)

    def ik(self, seed: np.ndarray, roll: float, target: np.ndarray, iters: int = 100) -> np.ndarray:
        self.k.robot.set_joint("wrist_roll", np.deg2rad(roll))
        q = np.asarray(seed, dtype=float)
        for _ in range(iters):            # placo's solve() is one step; iterate to convergence
            q = self.k.inverse_kinematics(q, target, orientation_weight=self.orientation_weight)
            if np.linalg.norm(self.fk(q, roll)[:3, 3] - target[:3, 3]) < 1e-5:
                break
        return q


def weights(n: int, a: int, b: int, c: int, e: int) -> np.ndarray:
    """0 before a, cosine up to 1 over a..b, 1 until c, cosine down to 0 over c..e."""
    w = np.zeros(n)
    for i in range(a, min(e, n)):
        if i < b:
            w[i] = 0.5 * (1 - np.cos(np.pi * (i - a) / (b - a)))
        elif i <= c:
            w[i] = 1.0
        else:
            w[i] = 0.5 * (1 + np.cos(np.pi * (i - c) / (e - c)))
    return w


def shifted(skill: dict, window: tuple[int, int, int, int], d: np.ndarray, kin: Kinematics):
    """(new frames, report rows). d in metres, base frame."""
    keys, frames = skill["keys"], skill["frames"]
    cols = [keys.index(f"{j}.pos") for j in SOLVE]
    roll_col = keys.index("wrist_roll.pos")
    w = weights(len(frames), *window)
    out = [list(r) for r in frames]
    report, seed = [], None
    for i in np.flatnonzero(w):
        q = np.array([frames[i][c] for c in cols])
        roll = frames[i][roll_col]
        T0 = kin.fk(q, roll)
        T = kin.target(T0, w[i] * d)
        sol = kin.ik(q if seed is None else seed, roll, T)
        got = kin.fk(sol, roll)
        residual = float(np.linalg.norm(got[:3, 3] - T[:3, 3]))
        if residual > MAX_RESIDUAL_M:
            raise RuntimeError(f"frame {i}: can't reach the shifted pose ({residual * 1000:.1f} mm off)")
        tilt = float(np.degrees(np.arccos(np.clip(got[:3, 2] @ T0[:3, 2], -1, 1))))
        for c, v in zip(cols, sol):
            out[i][c] = float(v)
        report.append((int(i), float(w[i]), (got[:3, 3] - T0[:3, 3]) * 1000, residual * 1000, tilt, sol - q))
        seed = sol
    return out, report


def max_step(frames: list[list[float]], cols: list[int]) -> float:
    """Largest joint change between consecutive frames, degrees."""
    a = np.array(frames)[:, cols]
    return float(np.abs(np.diff(a, axis=0)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skill")
    ap.add_argument("--dx", type=float, default=0.0, help="cm, base frame")
    ap.add_argument("--dy", type=float, default=0.0, help="cm, base frame")
    ap.add_argument("--dz", type=float, default=0.0, help="cm, base frame (negative = lower)")
    ap.add_argument("--window", type=int, nargs=4, metavar=("A", "B", "C", "E"), required=True)
    ap.add_argument("--dry-run", action="store_true", help="report, don't write")
    args = ap.parse_args()

    a, b, c, e = args.window
    if not a < b <= c < e:
        ap.error("--window needs A < B <= C < E")
    skill = arm.load_skill(args.skill)
    d = np.array([args.dx, args.dy, args.dz]) / 100
    frames, report = shifted(skill, (a, b, c, e), d, Kinematics())

    print(" frame     w   shift x/y/z mm         resid mm  tilt deg  joint change deg (pan lift elbow wflex)")
    for i, wi, s, r, tilt, dq in report[::max(1, len(report) // 20)] + report[-1:]:
        print(f"{i:6d} {wi:5.2f}  {s[0]:6.2f} {s[1]:6.2f} {s[2]:6.2f}   {r:7.3f}  {tilt:7.2f}   "
              + " ".join(f"{x:6.2f}" for x in dq))
    cols = [skill["keys"].index(f"{j}.pos") for j in SOLVE]
    print(f"largest joint step between frames: {max_step(skill['frames'], cols):.2f} deg before, "
          f"{max_step(frames, cols):.2f} deg after")

    if args.dry_run:
        return
    src = arm.skill_path(args.skill)
    backup = src.parent / "backup" / f"{args.skill}.{datetime.now():%Y%m%d-%H%M%S}.json"
    backup.parent.mkdir(exist_ok=True)
    shutil.copy2(src, backup)
    skill["frames"] = frames
    skill.setdefault("adjustments", []).append(
        {"at": datetime.now().isoformat(timespec="seconds"), "d_cm": [args.dx, args.dy, args.dz],
         "window": [a, b, c, e]})
    src.write_text(json.dumps(skill))
    print(f"wrote {src} (original in {backup})")


if __name__ == "__main__":
    main()
