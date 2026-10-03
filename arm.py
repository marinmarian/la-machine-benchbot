"""Thin wrapper around lerobot's SO-101 follower/leader + trajectory file format.

Trajectory file = JSON:
{
  "name": "screwdriver",
  "fps": 30,
  "keys": ["shoulder_pan.pos", ...],
  "frames": [[v0, v1, ...], ...]      # one row per tick, same order as keys
}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

def _load_dotenv(path: Path = Path(__file__).parent / ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ (existing vars win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

FOLLOWER_PORT = os.environ.get("FOLLOWER_PORT", "/dev/tty.usbmodem_FOLLOWER")
LEADER_PORT = os.environ.get("LEADER_PORT", "/dev/tty.usbmodem_LEADER")
FOLLOWER_ID = os.environ.get("FOLLOWER_ID", "bench_follower")  # must match lerobot-calibrate id
LEADER_ID = os.environ.get("LEADER_ID", "bench_leader")
# False (default) keeps the follower holding its pose after a script exits.
DISABLE_TORQUE_ON_DISCONNECT = os.environ.get("DISABLE_TORQUE_ON_DISCONNECT", "false").lower() in ("1", "true", "yes")

SKILLS_DIR = Path(__file__).parent / "skills"
SKILLS_DIR.mkdir(exist_ok=True)


def connect_follower(max_relative_target: float | None = None) -> SO101Follower:
    # max_relative_target caps how far a single command may move a joint, in degrees
    # (lerobot default use_degrees=True; gripper is 0..100). Safety net if a file is corrupt.
    cfg = SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID,
                              max_relative_target=max_relative_target,
                              disable_torque_on_disconnect=DISABLE_TORQUE_ON_DISCONNECT)
    robot = SO101Follower(cfg)
    robot.connect()
    return robot


def connect_leader() -> SO101Leader:
    teleop = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID))
    teleop.connect()
    return teleop


def current_pose(robot: SO101Follower) -> dict[str, float]:
    obs = robot.get_observation()
    return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}


def skill_path(name: str) -> Path:
    return SKILLS_DIR / f"{name}.json"


def save_skill(name: str, keys: list[str], frames: list[list[float]], fps: int) -> Path:
    p = skill_path(name)
    p.write_text(json.dumps({"name": name, "fps": fps, "keys": keys, "frames": frames}))
    return p


def load_skill(name: str) -> dict:
    return json.loads(skill_path(name).read_text())


def list_skills() -> list[str]:
    return sorted(p.stem for p in SKILLS_DIR.glob("*.json"))


GLIDE_DEG_PER_S = 30.0   # max joint speed while gliding to a start pose


def move_to(robot: SO101Follower, target: dict[str, float], seconds: float | None = None, fps: int = 30):
    """Linearly interpolate from the current pose to `target` so the arm never jumps.

    If `seconds` is None the duration is scaled to the largest joint delta
    (GLIDE_DEG_PER_S), with a 1.5 s floor, so far-away starts are not fast.
    """
    start = current_pose(robot)
    if seconds is None:
        biggest = max(abs(target[k] - start[k]) for k in target)
        seconds = max(1.5, biggest / GLIDE_DEG_PER_S)
    n = max(1, int(seconds * fps))
    for i in range(1, n + 1):
        a = i / n
        robot.send_action({k: start[k] + (target[k] - start[k]) * a for k in target})
        time.sleep(1 / fps)


def play(robot: SO101Follower, skill: dict, speed: float = 1.0, should_stop=lambda: False) -> bool:
    """Replay a skill. Returns False if interrupted."""
    keys, frames, fps = skill["keys"], skill["frames"], skill["fps"]
    speed = min(speed, 1.0)                       # never faster than it was recorded
    first = dict(zip(keys, frames[0]))
    move_to(robot, first)                         # glide to the start pose first (distance-scaled)
    dt = 1 / (fps * speed)
    t0 = time.perf_counter()
    for i, row in enumerate(frames):
        if should_stop():
            return False
        robot.send_action(dict(zip(keys, row)))
        # sleep against wall clock so timing doesn't drift
        target_t = t0 + (i + 1) * dt
        time.sleep(max(0.0, target_t - time.perf_counter()))
    return True
