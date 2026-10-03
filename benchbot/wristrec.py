"""Record wrist-camera frames + follower joint state, to try SAM tracking and fit the
pixels-per-degree table offline.

    python wristrec.py --probe                         # tile every camera with its index; find the wrist one
    python wristrec.py hover1 --camera 1               # teleop with the leader, keys below
    python wristrec.py screw1 --camera 1 --skill screwdriver --hold-at-hover

Keys (camera window focused):
    r  start/stop continuous recording       s  save one snapshot
    h  hold the follower where it is / resume (teleop: glides back to the leader; skill: pauses)
    j  jog probe from the held pose: each arm joint +-step, one settled frame per position
    c  continue the skill from where it stopped (--skill)
    q  quit

Typical session: drive to a hover pose above the bench, h to hold (hands are free now), move the
tool around under the camera with s / r, then j once for the table. A skill run with
--hold-at-hover stops at the same pose the grasp will start from.

Output: wrist_data/<session>/frames/*.jpg, samples.jsonl (one row per frame: t, file, mode, tag,
q = measured follower joints, cmd = last command), meta.json. Running an existing session appends.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import arm

OUT = Path(__file__).parent / "wrist_data"
JOG_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
HELP = "r rec  s snap  h hold  j jog  c continue  q quit"


class Camera:
    """Newest frame, its capture time and a counter, from a background thread."""

    def __init__(self, index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index} (find it with --probe)")
        self.frame, self.t, self.seq = None, 0.0, 0
        self._lock, self._stop = threading.Lock(), threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self.frame, self.t, self.seq = f, time.perf_counter(), self.seq + 1
        self.cap.release()

    def latest(self, timeout: float = 5.0):
        t = time.perf_counter()
        while self.frame is None and time.perf_counter() - t < timeout:
            time.sleep(0.02)
        if self.frame is None:
            raise RuntimeError("camera returned no frame")
        with self._lock:
            return self.frame, self.t, self.seq

    def close(self):
        self._stop.set()


class Writer:
    """Saves frames + rows off the control loop so JPEG encoding never stalls the arm."""

    def __init__(self, root: Path):
        self.root = root
        (root / "frames").mkdir(parents=True, exist_ok=True)
        self.n = len(list((root / "frames").glob("*.jpg")))
        self.dropped = 0
        self.rows = open(root / "samples.jsonl", "a")
        self.q: queue.Queue = queue.Queue(maxsize=256)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def save(self, frame: np.ndarray, row: dict) -> None:
        row = {"i": self.n, "file": f"frames/{self.n:06d}.jpg", **row}
        try:
            self.q.put_nowait((frame, row))
            self.n += 1
        except queue.Full:
            self.dropped += 1

    def _loop(self):
        while (item := self.q.get()) is not None:
            frame, row = item
            cv2.imwrite(str(self.root / row["file"]), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            self.rows.write(json.dumps(row) + "\n")
            self.rows.flush()

    def close(self):
        self.q.put(None)
        self.thread.join()
        self.rows.close()


def probe(max_index: int = 6) -> None:
    """Show one frame from every camera that opens, labelled with its index."""
    tiles = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        for _ in range(5):
            cap.grab()
        ok, f = cap.read()
        cap.release()
        if not ok:
            continue
        print(f"camera {i}: {f.shape[1]}x{f.shape[0]}")
        t = cv2.resize(f, (320, 240))
        cv2.putText(t, str(i), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        tiles.append(t)
    if not tiles:
        print("no cameras opened (macOS: give the terminal camera permission)")
        return
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
    grid = np.vstack([np.hstack(tiles[r:r + 3]) for r in range(0, len(tiles), 3)])
    cv2.imshow("cameras (any key to close)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_hover(skill: dict, lead_s: float) -> tuple[int, int]:
    """(grasp, hover) frame indices: grasp = the gripper first starts closing after opening,
    hover = lead_s seconds earlier."""
    g = np.array([r[skill["keys"].index("gripper.pos")] for r in skill["frames"]])
    opened = np.flatnonzero(g > g[0] + 15)
    if not opened.size:
        raise RuntimeError("gripper never opens in this skill; pass --hover-frame")
    run_max = g[opened[0]]
    for i in range(opened[0], len(g)):
        run_max = max(run_max, g[i])
        if g[i] < run_max - 8:
            return i, max(0, i - int(lead_s * skill["fps"]))
    raise RuntimeError("gripper never closes after opening; pass --hover-frame")


def glide(a: dict, b: dict) -> dict:
    """Motion segment from pose a to b, distance-scaled like arm.move_to."""
    biggest = max(abs(b[k] - a[k]) for k in b)
    return {"target": dict(b), "seconds": max(1.5, biggest / arm.GLIDE_DEG_PER_S)}


def jog_segments(base: dict, step: float) -> list[dict]:
    """Each arm joint +step and -step from base, holding still before every tagged frame."""
    segs = [{"target": base, "seconds": 0.5, "tag": "jog/base"}]
    for j in JOG_JOINTS:
        for d in (step, -step):
            tgt = {**base, f"{j}.pos": base[f"{j}.pos"] + d}
            segs += [{"target": tgt, "seconds": 0.6}, {"target": tgt, "seconds": 0.5, "tag": f"jog/{j}/{d:+g}"}]
        segs.append({"target": base, "seconds": 0.6})
    segs.append({"target": base, "seconds": 0.5, "tag": "jog/base"})
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?")
    ap.add_argument("--probe", action="store_true", help="show every camera with its index, then exit")
    ap.add_argument("--camera", type=int, default=os.environ.get("WRIST_CAMERA"),
                    help="wrist camera index (default: WRIST_CAMERA in .env)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30, help="control rate in teleop (skills use their own fps)")
    ap.add_argument("--skill", help="replay this skill instead of teleop (no leader needed)")
    ap.add_argument("--hold-at-hover", action="store_true", help="stop the skill at the hover frame")
    ap.add_argument("--hover-lead", type=float, default=1.0, help="hover = this many seconds before the grasp")
    ap.add_argument("--hover-frame", type=int, help="hover frame index, overrides the automatic one")
    ap.add_argument("--jog-step", type=float, default=3.0, help="degrees per jog")
    args = ap.parse_args()

    if args.probe:
        probe()
        return
    if not args.session:
        ap.error("session name required (or --probe)")
    if args.camera is None:
        ap.error("pass --camera N (find it with --probe) or set WRIST_CAMERA in .env")

    skill, hover = None, None
    if args.skill:
        skill = arm.load_skill(args.skill)
        if args.hover_frame is not None:
            hover = args.hover_frame
        elif args.hold_at_hover:
            grasp, hover = find_hover(skill, args.hover_lead)
            print(f"grasp starts at frame {grasp} ({grasp / skill['fps']:.1f}s), hover = frame {hover}")
    fps = skill["fps"] if skill else args.fps

    cam = Camera(int(args.camera), args.width, args.height)
    frame, _, _ = cam.latest()
    h, w = frame.shape[:2]
    robot = arm.connect_follower()
    leader = None if skill else arm.connect_leader()
    root = OUT / args.session
    writer = Writer(root)
    meta = {"session": args.session, "created": datetime.now().isoformat(timespec="seconds"),
            "camera": int(args.camera), "resolution": [w, h], "fps": fps, "follower_id": arm.FOLLOWER_ID,
            "skill": args.skill, "hover_frame": hover, "jog_step": args.jog_step}
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"camera {args.camera} at {w}x{h}, saving to {root}")

    def skill_pose(i: int) -> dict:
        return dict(zip(skill["keys"], skill["frames"][i]))

    cmd = arm.current_pose(robot)
    si = 0                                        # next skill frame
    motion: list[dict] = []                       # queued segments, run one tick at a time
    seg_t0, seg_from = None, None
    if skill:
        motion.append(glide(cmd, skill_pose(0)))
        mode, next_mode, recording = "moving", "skill", True
    else:
        motion.append(glide(cmd, leader.get_action()))
        mode, next_mode, recording = "moving", "teleop", False

    t_start = t0 = time.perf_counter()           # t_start stamps rows; t0 is the pacing clock
    last_seq, tick = 0, 0

    def save(tag: str | None) -> None:
        nonlocal last_seq
        frame, ft, seq = cam.latest()
        last_seq = seq
        writer.save(frame, {"t": round(ft - t_start, 4), "seq": seq, "mode": mode, "tag": tag,
                            "skill_frame": si if skill else None,
                            "q": arm.current_pose(robot), "cmd": cmd})

    try:
        while True:
            now = time.perf_counter()
            if motion:
                seg = motion[0]
                if seg_t0 is None:
                    seg_t0, seg_from = now, dict(cmd)
                a = min(1.0, (now - seg_t0) / seg["seconds"])
                cmd = {k: seg_from[k] + (seg["target"][k] - seg_from[k]) * a for k in seg["target"]}
                arm.send(robot, cmd)
                if a >= 1.0:
                    motion.pop(0)
                    seg_t0 = None
                    if seg.get("tag"):
                        save(seg["tag"])
                    if not motion:
                        mode = next_mode
            elif mode == "teleop":
                cmd = leader.get_action()
                arm.send(robot, cmd)
            elif mode == "skill":
                cmd = skill_pose(si)
                arm.send(robot, cmd)
                si += 1
                if args.hold_at_hover and si == hover:
                    mode = "hold"
                    print(f"holding at hover frame {hover}: move the tool around, s / r / j, c to continue")
                elif si >= len(skill["frames"]):
                    mode = "hold"
                    print("skill finished")
            # hold: send nothing, the motors keep the last goal

            frame, _, seq = cam.latest()
            if recording and seq != last_seq:
                save(None)

            view = frame.copy()
            cv2.drawMarker(view, (w // 2, h // 2), (0, 255, 255), cv2.MARKER_CROSS, 24, 1)
            state = f"{mode}" + (f"  frame {si}/{len(skill['frames'])}" if skill else "")
            cv2.putText(view, f"{state}   saved {writer.n}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(view, HELP, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if recording:
                cv2.circle(view, (w - 20, 20), 8, (0, 0, 255), -1)
            cv2.imshow("wrist camera", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                recording = not recording
            elif key == ord("s"):
                save("snap")
            elif key == ord("h") and not motion:
                if mode in ("teleop", "skill"):
                    mode = "hold"
                elif mode == "hold" and leader:
                    motion.append(glide(cmd, leader.get_action()))
                    mode, next_mode = "moving", "teleop"
            elif key == ord("c") and skill and mode == "hold" and not motion and si < len(skill["frames"]):
                motion.append(glide(cmd, skill_pose(si)))
                mode, next_mode = "moving", "skill"
            elif key == ord("j") and mode == "hold" and not motion:
                motion += jog_segments(cmd, args.jog_step)
                mode, next_mode = "jog", "hold"

            tick += 1
            target_t = t0 + tick / fps
            if target_t < time.perf_counter() - 0.5:   # fell far behind: don't sprint to catch up
                t0, tick = time.perf_counter(), 0
            time.sleep(max(0.0, target_t - time.perf_counter()))
    finally:
        cv2.destroyAllWindows()
        cam.close()
        writer.close()
        arm.disconnect(robot)
        if leader:
            leader.disconnect()
        print(f"{writer.n} frames in {root}" + (f" ({writer.dropped} dropped: disk too slow)" if writer.dropped else ""))


if __name__ == "__main__":
    main()
