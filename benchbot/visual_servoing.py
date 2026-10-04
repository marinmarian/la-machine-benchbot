"""Line the gripper up on the tool with the wrist camera, then grasp with the recorded skill.

    python visual_servoing.py screwdriver4 --fit-table screw4_2   # once: pixels-per-mm at the hover, from a wristrec jog session
    python visual_servoing.py screwdriver4 --set-target           # once: tool where the recording grasps it -> remember where it looks
    python visual_servoing.py screwdriver4 --align-only --show    # play to the hover, line up, stop there (arm keeps holding)
    python visual_servoing.py screwdriver4 --show                 # the whole thing

Flow: play the skill up to its "hover_frame" and stop. SAM 3 finds the tool by text ("prompt", default the
skill name minus digits) and SAM 2 tracks it from there. Each step moves the gripper in x/y at the hover's
height and tilt (inverse kinematics; pan turns it as a sideways move must) by gain * table^-1 * (target - centroid), lets it settle and
measures again, until the centroid is within --tol px of "hover_target". The rest of the skill then plays
with that whole x/y offset from the hover on, and eases back to the recorded path over the "rejoin"
frames [c, e] (during the lift), so the end of the skill is unchanged. Position only: the tool must lie at
roughly the recorded angle (README).

Skill keys used: hover_frame, rejoin, hover_target, hover_px_per_mm, prompt (optional).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import cv2
import numpy as np

import arm
from adjust import SOLVE, Kinematics, shifted
from wristrec import OUT, Camera


def rotvec(R: np.ndarray) -> np.ndarray:
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if a < 1e-9:
        return np.zeros(3)
    return a / (2 * np.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def joints(q: dict) -> np.ndarray:
    return np.array([q[f"{j}.pos"] for j in SOLVE])


def fit_table(session: str, prompt: str, kin: Kinematics) -> tuple[np.ndarray, dict]:
    """2x2 pixels per mm of gripper x/y motion (base frame) at constant height and tilt, from jog frames.

    A raw jog also tilts and lifts the camera, which moves the image by itself. Lift, elbow and
    wrist_flex all move the gripper along the same radial line, so their three pixel shifts are split
    into radial motion + height + tilt (3 unknowns per image axis). Sideways motion always comes with
    the pan rotation (there is no other yaw joint), so pan's shift is used as it is.
    """
    import samseg
    root = OUT / session
    rows = [json.loads(l) for l in open(root / "samples.jsonl")]
    jog = [r for r in rows if (r["tag"] or "").startswith("jog/") and r["tag"] != "jog/base"]
    sam3 = samseg.load_sam3()
    by = {}                                               # joint -> sign -> [(centroid, q)]
    for r in jog:
        _, j, step = r["tag"].split("/")
        res = samseg.find_text_mask(sam3, cv2.imread(str(root / r["file"])), prompt)
        if res is None:
            print(f"  frame {r['i']} {r['tag']}: no '{prompt}' found, skipped")
            continue
        by.setdefault(j, {}).setdefault(step[0], []).append((samseg.mask_centroid(res[0]), r["q"]))
    T0 = None
    stats = {}
    for j in SOLVE:
        pairs = list(zip(by[j]["+"], by[j]["-"]))
        if not pairs:
            raise RuntimeError(f"no usable +/- jog pair for {j}")
        if T0 is None:
            q = pairs[0][1][1]
            T0 = kin.fk(joints(q), q["wrist_roll.pos"])
        px, dp, rv = np.zeros(2), np.zeros(3), np.zeros(3)
        for (cp, qp), (cm, qm) in pairs:
            Tp, Tm = kin.fk(joints(qp), qp["wrist_roll.pos"]), kin.fk(joints(qm), qm["wrist_roll.pos"])
            dq = qp[f"{j}.pos"] - qm[f"{j}.pos"]
            px += (np.array(cp) - np.array(cm)) / dq
            dp += (Tp[:3, 3] - Tm[:3, 3]) * 1000 / dq
            rv += np.degrees(rotvec(Tp[:3, :3] @ Tm[:3, :3].T)) / dq
        stats[j] = (px / len(pairs), dp / len(pairs), rv / len(pairs), len(pairs))
    p0 = T0[:3, 3]
    rhat = np.r_[p0[:2] / np.linalg.norm(p0[:2]), 0]
    that = np.array([-rhat[1], rhat[0], 0])
    pitch = np.cross([0, 0, 1], rhat)
    for j, (px, dp, rv, n) in stats.items():
        print(f"  {j:14s} {n} pairs  px/deg ({px[0]:6.1f},{px[1]:6.1f})  radial {dp @ rhat:5.2f} "
              f"side {dp @ that:5.2f} z {dp[2]:5.2f} mm/deg  tilt {rv @ pitch:5.2f} deg/deg")
    tilt_joints = ["shoulder_lift", "elbow_flex", "wrist_flex"]
    A = np.array([[stats[j][1] @ rhat, stats[j][1][2], stats[j][2] @ pitch] for j in tilt_joints])
    radial = np.linalg.solve(A, np.array([stats[j][0] for j in tilt_joints]))[0]
    side = stats["shoulder_pan"][0] / (stats["shoulder_pan"][1] @ that)
    B = np.c_[radial, side] @ np.array([rhat[:2], that[:2]])   # columns: base x, base y
    return B, {"radial": radial.tolist(), "side": side.tolist(), "cond": float(np.linalg.cond(A))}


def default_prompt(name: str) -> str:
    return re.sub(r"\d+$", "", name).replace("_", " ")


class Aligner:
    def __init__(self, cam: Camera, prompt: str, show: bool):
        import samseg
        self.samseg, self.cam, self.prompt, self.show = samseg, cam, prompt, show
        t = time.perf_counter()
        self.sam3 = samseg.load_sam3()
        self.tracker = samseg.MaskTracker("tiny")
        print(f"SAM 3 + SAM 2 loaded in {time.perf_counter() - t:.0f}s")
        self.primed = False

    def fresh_frame(self, after: float) -> np.ndarray:
        """First camera frame captured after time `after` (perf_counter)."""
        while True:
            frame, t, _ = self.cam.latest()
            if t > after:
                return frame
            time.sleep(0.01)

    def detect(self, frame: np.ndarray):
        res = self.samseg.find_text_mask(self.sam3, frame, self.prompt)
        if res is None:
            return None
        self.tracker.prime(frame, res[0])
        self.primed = True
        return res[0]

    def measure(self, settle: float = 0.4, target=None):
        """Tool centroid (u, v) on a settled frame, or None if it isn't in view."""
        time.sleep(settle)
        frame = self.fresh_frame(time.perf_counter())
        mask = self.tracker.track(frame) if self.primed else None
        if mask is None or not mask.any():
            mask = self.detect(frame)                       # first look, or tracking lost: ask SAM 3
        c = None if mask is None else self.samseg.mask_centroid(mask)
        if self.show:
            vis = frame.copy()
            if mask is not None:
                vis[mask] = (0.5 * vis[mask] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
            if target is not None:
                cv2.drawMarker(vis, (int(target[0]), int(target[1])), (255, 255, 0), cv2.MARKER_CROSS, 30, 2)
            if c is not None:
                cv2.circle(vis, (int(c[0]), int(c[1])), 6, (0, 0, 255), -1)
            cv2.imshow("visual servoing", vis)
            cv2.waitKey(1)
        return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skill")
    ap.add_argument("--prompt", help="what SAM 3 looks for (default: skill's 'prompt', else its name without digits)")
    ap.add_argument("--camera", type=int, default=os.environ.get("WRIST_CAMERA"))
    ap.add_argument("--fit-table", metavar="SESSION", help="fit hover_px_per_mm from a wristrec jog session, save it, exit")
    ap.add_argument("--set-target", action="store_true", help="play to the hover, save where the tool looks as hover_target, exit")
    ap.add_argument("--align-only", action="store_true", help="stop after lining up (arm keeps holding)")
    ap.add_argument("--tol", type=float, default=6.0, help="px; done when the centroid is this close (~1.5 mm)")
    ap.add_argument("--gain", type=float, default=0.7)
    ap.add_argument("--max-step", type=float, default=15.0, help="mm per correction")
    ap.add_argument("--max-offset", type=float, default=40.0, help="mm; give up beyond this")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--show", action="store_true", help="window with the mask, centroid (red) and target (cyan)")
    args = ap.parse_args()

    skill = arm.load_skill(args.skill)
    prompt = args.prompt or skill.get("prompt") or default_prompt(args.skill)
    kin = Kinematics()

    if args.fit_table:
        B, info = fit_table(args.fit_table, prompt, kin)
        print(f"px per mm (rows u, v; columns base x, y):\n{B.round(2)}\n{info}")
        skill["hover_px_per_mm"] = B.tolist()
        arm.skill_path(args.skill).write_text(json.dumps(skill))
        print(f"saved hover_px_per_mm in {args.skill}")
        return

    hover = skill.get("hover_frame")
    if hover is None:
        raise SystemExit(f"{args.skill} has no hover_frame")
    if not args.set_target:
        missing = [k for k in ("hover_target", "hover_px_per_mm") + (() if args.align_only else ("rejoin",))
                   if k not in skill]
        if missing:
            raise SystemExit(f"{args.skill} is missing {', '.join(missing)} (see the docstring for how to set them)")
    if args.camera is None:
        raise SystemExit("pass --camera N or set WRIST_CAMERA in .env")

    cam = Camera(int(args.camera), 640, 480)
    cam.latest()
    aligner = Aligner(cam, prompt, args.show)
    robot = arm.connect_follower()
    try:
        keys = skill["keys"]
        arm.play(robot, {**skill, "frames": skill["frames"][:hover + 1]})
        q_h = dict(zip(keys, skill["frames"][hover]))
        roll = q_h["wrist_roll.pos"]
        T_h = kin.fk(joints(q_h), roll)

        if args.set_target:
            cs = [aligner.measure(settle=0.8 if k == 0 else 0.2) for k in range(5)]
            cs = [c for c in cs if c is not None]
            if not cs:
                raise SystemExit(f"no '{prompt}' in view at the hover")
            skill["hover_target"] = np.mean(cs, axis=0).round(1).tolist()
            arm.skill_path(args.skill).write_text(json.dumps(skill))
            print(f"saved hover_target {skill['hover_target']} (spread {np.ptp(cs, axis=0).round(1)} px)")
            return

        target = np.array(skill["hover_target"])
        Binv = np.linalg.inv(np.array(skill["hover_px_per_mm"]))
        D = np.zeros(3)                                    # mm, base frame, z stays 0
        q = joints(q_h)
        ok = False
        for it in range(args.iters + 1):
            c = aligner.measure(settle=0.8 if it == 0 else 0.4, target=target)
            if c is None:
                raise SystemExit(f"no '{prompt}' in view; nothing moved past the hover")
            e = target - np.array(c)
            print(f"  step {it}: centroid ({c[0]:.0f}, {c[1]:.0f})  error {np.linalg.norm(e):5.1f} px  "
                  f"offset x {D[0]:5.1f} y {D[1]:5.1f} mm")
            if np.linalg.norm(e) <= args.tol:
                ok = True
                break
            if it == args.iters:
                break
            step = args.gain * (Binv @ e)
            if np.linalg.norm(step) > args.max_step:
                step *= args.max_step / np.linalg.norm(step)
            D[:2] += step
            if np.linalg.norm(D) > args.max_offset:
                raise SystemExit(f"needs more than {args.max_offset:.0f} mm; is the tool where the skill expects it?")
            q = kin.ik(q, roll, kin.target(T_h, D / 1000))
            arm.move_to(robot, {**q_h, **{f"{j}.pos": float(v) for j, v in zip(SOLVE, q)}}, seconds=0.6)
        if not ok:
            raise SystemExit(f"didn't converge in {args.iters} steps; stopping at the hover")
        print(f"lined up: offset x {D[0]:.1f} y {D[1]:.1f} mm")
        if args.align_only:
            return

        c_, e_ = skill["rejoin"]
        frames, _ = shifted(skill, (hover - 1, hover, c_, e_), D / 1000, kin)
        arm.play(robot, {**skill, "frames": frames[hover:]})
    finally:
        cam.close()
        arm.disconnect(robot)


if __name__ == "__main__":
    main()
