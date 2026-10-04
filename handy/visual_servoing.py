"""Line the gripper up on the tool with the wrist camera, then grasp with the recorded skill.

    python visual_servoing.py screwdriver4 --fit-table screw4_2   # once: pixels-per-mm at the hover, from a wristrec session (k or j probe)
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
from wristrec import OUT, Camera, CameraLost


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
    by = {j: [] for j in SOLVE}                          # joint -> [(i, sign, centroid, q)] in recording order
    for r in jog:
        _, j, step = r["tag"].split("/")
        frame = cv2.imread(str(root / r["file"]))
        # a settled frame we picked ourselves: a weaker detection is still the object
        res = samseg.find_text_mask(sam3, frame, prompt) or samseg.find_text_mask(sam3, frame, prompt, threshold=0.3)
        if res is None:
            print(f"  frame {r['i']} {r['tag']}: no '{prompt}' found, skipped")
            continue
        by[j].append((r["i"], step[0], samseg.mask_centroid(res[0]), r["q"]))
    T0 = None
    stats = {}
    for j in SOLVE:
        # each jog run saves +step then -step; pair them within a run (the tool may move between runs)
        seq = sorted(by[j], key=lambda x: x[0])
        pairs = [((a[2], a[3]), (b[2], b[3])) for a, b in zip(seq, seq[1:]) if (a[1], b[1]) == ("+", "-")]
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


def wait_still(rig, max_s: float = 0.8, tol: float = 0.2, dt: float = 0.05) -> None:
    """Return once no arm joint moves more than `tol` degrees in `dt`, or after `max_s`.

    A hover inside a pause in the recording is already still on arrival (black_motor2), so this
    returns at once; a hover mid-motion (screwdriver4) waits for the arm to stop.
    """
    t0 = time.perf_counter()
    prev = arm.current_pose(rig)
    while time.perf_counter() - t0 < max_s:
        time.sleep(dt)
        cur = arm.current_pose(rig)
        if max(abs(cur[k] - prev[k]) for k in cur if not k.endswith("gripper.pos")) < tol:
            return
        prev = cur


def fit_slide_table(session: str, prompt: str, kin: Kinematics):
    """2x2 pixels per mm straight from wristrec's slide probe (k), or None if the session has none.

    The probe moves the gripper +-d mm in base x and y with the line-up's own inverse kinematics, so the
    table maps a commanded offset to the pixel shift it really produces (pan's turn and the joints'
    sag included). Least squares over all distances: centroid = c0[run] + B @ D. Each run gets its own
    c0 rather than its slide/base frame, because the joints' backlash makes the arm settle a few mm
    differently depending on the direction it arrived from. Masks touching the image border are left
    out: a half-visible tool's centroid is biased.
    """
    import samseg
    root = OUT / session
    rows = [json.loads(l) for l in open(root / "samples.jsonl")]
    rows = [r for r in rows if (r["tag"] or "").startswith("slide/")]
    if not rows:
        return None
    sam3 = samseg.load_sam3()
    pts, run, prev = [], 0, None                           # (run, axis, mm, D, centroid, gripper xy mm)
    for r in rows:
        if r["tag"] == "slide/base":
            if prev is not None and prev != "slide/base":
                run += 1
            prev = r["tag"]
            continue
        prev = r["tag"]
        frame = cv2.imread(str(root / r["file"]))
        res = samseg.find_text_mask(sam3, frame, prompt) or samseg.find_text_mask(sam3, frame, prompt, threshold=0.3)
        if res is None:
            print(f"  frame {r['i']} {r['tag']}: not found, skipped")
            continue
        m = res[0]
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            print(f"  frame {r['i']} {r['tag']}: touches the image edge, skipped")
            continue
        _, axis, mm = r["tag"].split("/")
        D = np.array([float(mm), 0.0]) if axis == "x" else np.array([0.0, float(mm)])
        xy = kin.fk(joints(r["q"]), r["q"]["wrist_roll.pos"])[:2, 3] * 1000
        pts.append((run, axis, float(mm), D, np.array(samseg.mask_centroid(m)), xy))
    if not {"x", "y"} <= {p[1] for p in pts}:
        raise RuntimeError("need usable slide frames along both x and y")
    runs = sorted({p[0] for p in pts})
    X = np.array([np.r_[p[3], [p[0] == k for k in runs]] for p in pts], dtype=float)
    C = np.array([p[4] for p in pts])
    sol = np.linalg.lstsq(X, C, rcond=None)[0]
    B = sol[:2].T
    resid = C - X @ sol
    A = np.linalg.lstsq(X, np.array([p[5] for p in pts]), rcond=None)[0][:2].T   # achieved mm per commanded mm
    print(f"  {len(pts)} slide frames in {len(runs)} runs, fit residual {np.sqrt((resid ** 2).sum(1).mean()):.1f} px rms; "
          f"arm moved {A[0, 0] * 100:.0f}% of the command in x, {A[1, 1] * 100:.0f}% in y")
    for axis in "xy":                                      # does the table drift with distance?
        for mm in sorted({abs(p[2]) for p in pts if p[1] == axis}):
            sel = [i for i, p in enumerate(pts) if p[1] == axis and abs(p[2]) == mm]
            local = np.mean([(C[i] - X[i, 2:] @ sol[2:]) / pts[i][2] for i in sel], axis=0)
            print(f"  {axis} {mm:3.0f} mm: {len(sel)} frames  px per mm ({local[0]:5.2f}, {local[1]:5.2f})  "
                  f"residual {np.sqrt((resid[sel] ** 2).sum(1).mean()):.1f} px")
    return B


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
        self.warm_up()
        self.primed = False

    def warm_up(self) -> None:
        """Run both models once now: the first call on the GPU is ~0.9 s slower, and nobody waits yet."""
        t = time.perf_counter()
        frame, _, _ = self.cam.latest()
        res = self.samseg.find_text_mask(self.sam3, frame, self.prompt)
        if res is None:                                     # parked pose: probably nothing in view
            mask = np.zeros(frame.shape[:2], bool)
            mask[200:280, 280:360] = True
        else:
            mask = res[0]
        self.tracker.prime(frame, mask)
        self.tracker.track(frame)
        print(f"warmed up in {time.perf_counter() - t:.1f}s")

    def fresh_frame(self, after: float, timeout: float = 2.0) -> np.ndarray:
        """First camera frame captured after time `after` (perf_counter)."""
        while True:
            frame, t, _ = self.cam.latest()
            if t > after:
                return frame
            if time.perf_counter() - after > timeout:
                raise CameraLost(f"camera {self.cam.index} stopped sending frames")
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


def prompt_for(name: str, skill: dict) -> str:
    return skill.get("prompt") or default_prompt(name)


class ServoFailed(Exception):
    """Lining up gave up (tool not seen, out of reach, error growing...). The arm went home first."""


class Stopped(Exception):
    """should_stop() fired: the arm holds where it is."""


def go_home(rig, a: str, skill: dict, why: str, should_stop=lambda: False) -> None:
    """Never leave the arm hanging over the bench: glide back to the recorded hover, then retrace the
    approach to the skill's start pose."""
    print(f"stopped: {why}\ngoing home")
    hover = skill["hover_frame"]
    arm.play(rig, arm.on(a, {**skill, "frames": skill["frames"][hover::-1]}), should_stop=should_stop)


def servo_grasp(rig, a: str, skill: dict, aligner: Aligner, kin: Kinematics, *, tol: float = 6.0,
                gain: float = 0.7, max_step: float = 20.0, max_offset: float = 100.0, iters: int = 12,
                align_only: bool = False, should_stop=lambda: False) -> np.ndarray:
    """Play `skill` (one arm's part, plain keys: arm.solo) on arm `a` to its hover, line up on the tool,
    then play the grasp with that offset. Returns the offset in mm. Any give-up after the hover sends the
    arm home and raises ServoFailed; should_stop() raises Stopped and leaves the arm where it is."""
    hover = skill["hover_frame"]
    aligner.primed = False                                 # a new run: find the tool afresh with SAM 3
    if not arm.play(rig, arm.on(a, {**skill, "frames": skill["frames"][:hover + 1]}), should_stop=should_stop):
        raise Stopped()
    try:
        D, frames = _line_up(rig, a, skill, aligner, kin, tol, gain, max_step, max_offset, iters, should_stop)
    except (ServoFailed, CameraLost) as e:
        go_home(rig, a, skill, str(e), should_stop)
        raise ServoFailed(str(e)) from e
    if align_only:
        return D
    if not arm.play(rig, arm.on(a, {**skill, "frames": frames[hover:]}), should_stop=should_stop):
        raise Stopped()
    return D


def _line_up(rig, a, skill, aligner, kin, tol, gain, max_step, max_offset, iters, should_stop):
    """Step at the hover's height until the tool sits on hover_target. (offset mm, shifted frames)."""
    keys, hover = skill["keys"], skill["hover_frame"]
    q_h = dict(zip(keys, skill["frames"][hover]))
    target = np.array(skill["hover_target"])
    Binv = np.linalg.inv(np.array(skill["hover_px_per_mm"]))
    window = (hover - 1, hover, *skill["rejoin"])
    cols = [keys.index(f"{j}.pos") for j in SOLVE]
    D = np.zeros(3)                                        # mm, base frame, z stays 0
    frames = skill["frames"]                               # the grasp for the current D
    errors = []
    for it in range(iters + 1):
        if should_stop():
            raise Stopped()
        if it == 0:
            wait_still(rig)
        c = aligner.measure(settle=0.0 if it == 0 else 0.4, target=target)
        if c is None:
            raise ServoFailed(f"no '{aligner.prompt}' in view")
        e = target - np.array(c)
        print(f"  step {it}: centroid ({c[0]:.0f}, {c[1]:.0f})  error {np.linalg.norm(e):5.1f} px  "
              f"offset x {D[0]:5.1f} y {D[1]:5.1f} mm")
        if np.linalg.norm(e) <= tol:
            print(f"lined up: offset x {D[0]:.1f} y {D[1]:.1f} mm")
            return D, frames
        errors.append(np.linalg.norm(e))
        if len(errors) >= 3 and errors[-1] > errors[-2] > errors[-3]:
            raise ServoFailed("the error grew twice in a row: wrong object, or the pixel table is off")
        if it == iters:
            break
        step = gain * (Binv @ e)
        if np.linalg.norm(step) > max_step:
            step *= max_step / np.linalg.norm(step)
        nxt = D.copy()
        nxt[:2] += step
        if np.linalg.norm(nxt) > max_offset:
            raise ServoFailed(f"needs more than {max_offset:.0f} mm; is the tool where the skill expects it?")
        try:                                               # the whole offset grasp must be reachable, not just this pose
            frames, _ = shifted(skill, window, nxt / 1000, kin)
        except RuntimeError as ex:
            raise ServoFailed(f"tool is out of reach at offset x {nxt[0]:.0f} y {nxt[1]:.0f} mm ({ex})")
        D = nxt
        arm.move_to(rig, arm.prefix(a, {**q_h, **{keys[c]: frames[hover][c] for c in cols}}), seconds=0.6)
    raise ServoFailed(f"didn't converge in {iters} steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skill")
    ap.add_argument("--prompt", help="what SAM 3 looks for (default: skill's 'prompt', else its name without digits)")
    ap.add_argument("--camera", type=int, default=os.environ.get("WRIST_CAMERA"))
    ap.add_argument("--arm", default=arm.DEFAULT_ARM, choices=arm.ARMS, help="which arm carries the wrist camera")
    ap.add_argument("--fit-table", metavar="SESSION", help="fit hover_px_per_mm from a wristrec jog session, save it, exit")
    ap.add_argument("--set-target", action="store_true", help="play to the hover, save where the tool looks as hover_target, exit")
    ap.add_argument("--align-only", action="store_true", help="stop after lining up (arm keeps holding)")
    ap.add_argument("--tol", type=float, default=6.0, help="px; done when the centroid is this close (~1.5 mm)")
    ap.add_argument("--gain", type=float, default=0.7)
    ap.add_argument("--max-step", type=float, default=20.0, help="mm per correction")
    ap.add_argument("--max-offset", type=float, default=100.0,
                    help="mm; give up beyond this (the reach of the whole shifted grasp is checked every step too)")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--show", action="store_true", help="window with the mask, centroid (red) and target (cyan)")
    args = ap.parse_args()

    full = arm.load_skill(args.skill)                      # written back with any new keys
    skill = arm.solo(full, args.arm)                       # this arm's joints, plain keys
    prompt = args.prompt or prompt_for(args.skill, skill)
    kin = Kinematics()

    if args.fit_table:
        B = fit_slide_table(args.fit_table, prompt, kin)
        if B is None:                                      # older session: per-joint jogs only
            B, info = fit_table(args.fit_table, prompt, kin)
            print(info)
        print(f"px per mm (rows u, v; columns base x, y):\n{B.round(2)}")
        full["hover_px_per_mm"] = B.tolist()
        arm.update_skill(args.skill, full)
        print(f"saved hover_px_per_mm in {args.skill}")
        return

    hover = skill.get("hover_frame")
    if hover is None:
        raise SystemExit(f"{args.skill} has no hover_frame")
    if not args.set_target:
        missing = [k for k in ("hover_target", "hover_px_per_mm", "rejoin") if k not in skill]
        if missing:
            raise SystemExit(f"{args.skill} is missing {', '.join(missing)} (see the docstring for how to set them)")
    if args.camera is None:
        raise SystemExit("pass --camera N or set WRIST_CAMERA in .env")

    cam = Camera(int(args.camera), 640, 480)
    cam.latest()
    aligner = Aligner(cam, prompt, args.show)
    a = args.arm
    rig = {a: arm.connect_follower(a)}
    try:
        if args.set_target:
            arm.play(rig, arm.on(a, {**skill, "frames": skill["frames"][:hover + 1]}))
            cs = [aligner.measure(settle=0.8 if k == 0 else 0.2) for k in range(5)]
            cs = [c for c in cs if c is not None]
            if not cs:
                go_home(rig, a, skill, f"no '{prompt}' in view at the hover")
                raise SystemExit(1)
            full["hover_target"] = np.mean(cs, axis=0).round(1).tolist()
            arm.update_skill(args.skill, full)
            print(f"saved hover_target {full['hover_target']} (spread {np.ptp(cs, axis=0).round(1)} px)")
            return
        servo_grasp(rig, a, skill, aligner, kin, tol=args.tol, gain=args.gain, max_step=args.max_step,
                    max_offset=args.max_offset, iters=args.iters, align_only=args.align_only)
    except ServoFailed:
        raise SystemExit(1)
    finally:
        cam.close()
        arm.disconnect(rig)


if __name__ == "__main__":
    main()
