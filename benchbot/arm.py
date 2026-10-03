"""Thin wrapper around lerobot's SO-101 follower/leader + trajectory file format.

Several arms can be driven together. Each arm has a name (default: "right", "left");
every joint key is prefixed with it, so a two-arm skill looks like:

{
  "name": "handoff",
  "fps": 30,
  "keys": ["right_shoulder_pan.pos", ..., "left_shoulder_pan.pos", ...],
  "frames": [[v0, v1, ...], ...]      # one row per tick, same order as keys
}

Older single-arm files have unprefixed keys ("shoulder_pan.pos"); load_skill() maps those
to the FIRST arm listed in ARMS, so they keep playing unchanged.

A "rig" is just a dict {arm_name: SO101Follower}; leaders are {arm_name: SO101Leader}.
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

# Arm names, in order. The first one is the "default" arm: legacy unprefixed skills play on it.
ARMS: list[str] = [a.strip() for a in os.environ.get("ARMS", "right").split(",") if a.strip()]
DEFAULT_ARM = ARMS[0]

# False (default) keeps the follower holding its pose after a script exits.
DISABLE_TORQUE_ON_DISCONNECT = os.environ.get("DISABLE_TORQUE_ON_DISCONNECT", "false").lower() in ("1", "true", "yes")

SKILLS_DIR = Path(__file__).parent / "skills"
SKILLS_DIR.mkdir(exist_ok=True)

GLIDE_DEG_PER_S = 30.0   # max joint speed while gliding to a start pose
GRIPPER_MIN = float(os.environ.get("GRIPPER_MIN", "3"))   # never command a gripper below this (0..100);
                                                          # grinding against the closed stop trips overload


def _env(arm: str, key: str) -> str:
    """Per-arm setting, e.g. _env("left", "FOLLOWER_PORT") -> $LEFT_FOLLOWER_PORT."""
    name = f"{arm.upper()}_{key}"
    try:
        return os.environ[name]
    except KeyError:
        raise RuntimeError(f"{name} is not set; add it to .env (arms configured: {', '.join(ARMS)})") from None


def parse_arms(spec: str | None) -> list[str]:
    """'left,right' -> ['left', 'right']; None/'' -> all configured arms. Validates names."""
    arms = [a.strip() for a in spec.split(",") if a.strip()] if spec else list(ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; ARMS in .env = {ARMS}")
    return arms


# ---------- connecting ----------

def _connect_with_file_calibration(device) -> None:
    """connect() without the interactive calibration prompt.

    If the motors' stored offsets/limits differ from the calibration file, write the
    file values to the motors (what pressing Enter at lerobot's prompt does). Never
    starts a new calibration: that needs a human and lerobot-calibrate.
    """
    if not device.calibration:
        raise RuntimeError(f"No calibration file for id {device.id!r}; run lerobot-calibrate first.")
    device.connect(calibrate=False)
    if not device.is_calibrated:
        print(f"{device.id}: motor calibration differs from file, writing file values to motors")
        # Lock=0 lets the Feetech motors persist these registers to EEPROM (lerobot sets
        # Lock=1 whenever torque is enabled, which would make the write RAM-only).
        for motor in device.bus.motors:
            device.bus.write("Lock", motor, 0)
        device.bus.write_calibration(device.calibration)
        for motor in device.bus.motors:
            device.bus.write("Lock", motor, 1)


def connect_follower(arm: str, max_relative_target: float | None = None) -> SO101Follower:
    # max_relative_target caps how far a single command may move a joint, in degrees
    # (lerobot default use_degrees=True; gripper is 0..100). Safety net if a file is corrupt.
    cfg = SO101FollowerConfig(port=_env(arm, "FOLLOWER_PORT"), id=_env(arm, "FOLLOWER_ID"),
                              max_relative_target=max_relative_target,
                              disable_torque_on_disconnect=DISABLE_TORQUE_ON_DISCONNECT)
    robot = SO101Follower(cfg)
    _connect_with_file_calibration(robot)
    return robot


def connect_leader(arm: str) -> SO101Leader:
    teleop = SO101Leader(SO101LeaderConfig(port=_env(arm, "LEADER_PORT"), id=_env(arm, "LEADER_ID")))
    _connect_with_file_calibration(teleop)
    return teleop


def connect_followers(arms: list[str] | None = None) -> dict[str, SO101Follower]:
    """Connect the followers of the given arms (default: all). On failure, disconnect
    the ones already up so no arm is left half-configured."""
    rig: dict[str, SO101Follower] = {}
    try:
        for a in arms or ARMS:
            try:
                rig[a] = connect_follower(a)
            except RuntimeError as e:
                others = [x for x in (arms or ARMS) if x != a]
                hint = f" To run without it: --arms {','.join(others)}" if others else ""
                raise RuntimeError(f"arm '{a}': {e}{hint}") from None
    except Exception:
        disconnect(rig)
        raise
    return rig


def connect_leaders(arms: list[str] | None = None) -> dict[str, SO101Leader]:
    leaders: dict[str, SO101Leader] = {}
    try:
        for a in arms or ARMS:
            leaders[a] = connect_leader(a)
    except Exception:
        for lead in leaders.values():
            lead.disconnect()
        raise
    return leaders


def disconnect(rig: dict[str, SO101Follower]) -> None:
    """Release each gripper (so it never sits pushing on its stop), then disconnect.
    Arm joints keep holding unless DISABLE_TORQUE_ON_DISCONNECT=true."""
    for robot in rig.values():
        try:
            robot.bus.disable_torque("gripper")
        finally:
            robot.disconnect()


# ---------- prefixed poses / actions ----------

def prefix(arm: str, d: dict[str, float]) -> dict[str, float]:
    return {f"{arm}_{k}": float(v) for k, v in d.items()}


def split(action: dict[str, float], arms) -> dict[str, dict[str, float]]:
    """{'left_x.pos': v} -> {'left': {'x.pos': v}}; keys for arms not in `arms` are ignored."""
    out: dict[str, dict[str, float]] = {}
    for key, v in action.items():
        for a in arms:
            if key.startswith(a + "_"):
                out.setdefault(a, {})[key[len(a) + 1:]] = v
                break
    return out


def current_pose(rig: dict[str, SO101Follower]) -> dict[str, float]:
    pose: dict[str, float] = {}
    for a, robot in rig.items():
        obs = robot.get_observation()
        pose.update(prefix(a, {k: v for k, v in obs.items() if k.endswith(".pos")}))
    return pose


def leader_action(leaders: dict[str, SO101Leader]) -> dict[str, float]:
    """Read every leader (back to back, so one frame's samples share a timestamp)."""
    action: dict[str, float] = {}
    for a, lead in leaders.items():
        action.update(prefix(a, lead.get_action()))
    return action


def send(rig: dict[str, SO101Follower], action: dict[str, float]) -> dict[str, float]:
    """send_action per arm with every gripper clamped to >= GRIPPER_MIN. Use this everywhere.
    Arms in `action` that are not in `rig` are skipped (so a two-arm skill still plays on one arm)."""
    sent: dict[str, float] = {}
    for a, part in split(action, rig).items():
        if "gripper.pos" in part and part["gripper.pos"] < GRIPPER_MIN:
            part = {**part, "gripper.pos": GRIPPER_MIN}
        sent.update(prefix(a, rig[a].send_action(part)))
    return sent


# ---------- skill files ----------

def skill_path(name: str) -> Path:
    return SKILLS_DIR / f"{name}.json"


def save_skill(name: str, keys: list[str], frames: list[list[float]], fps: int) -> Path:
    p = skill_path(name)
    p.write_text(json.dumps({"name": name, "fps": fps, "keys": keys, "frames": frames}))
    return p


def load_skill(name: str) -> dict:
    skill = json.loads(skill_path(name).read_text())
    # legacy single-arm file: keys like "shoulder_pan.pos" -> "<default arm>_shoulder_pan.pos"
    skill["keys"] = [k if any(k.startswith(a + "_") for a in ARMS) else f"{DEFAULT_ARM}_{k}"
                     for k in skill["keys"]]
    return skill


def skill_arms(skill: dict) -> list[str]:
    """Which arms a (loaded) skill moves, in ARMS order."""
    return [a for a in ARMS if any(k.startswith(a + "_") for k in skill["keys"])]


def list_skills() -> list[str]:
    return sorted(p.stem for p in SKILLS_DIR.glob("*.json"))


# ---------- motion ----------

def move_to(rig: dict[str, SO101Follower], target: dict[str, float], seconds: float | None = None, fps: int = 30):
    """Linearly interpolate from the current pose to `target` so no arm ever jumps.

    If `seconds` is None the duration is scaled to the largest joint delta over all arms
    (GLIDE_DEG_PER_S), with a 1.5 s floor, so far-away starts are not fast.
    """
    start = current_pose(rig)
    target = {k: v for k, v in target.items() if k in start}      # ignore arms not connected
    if not target:
        return
    if seconds is None:
        biggest = max(abs(target[k] - start[k]) for k in target)
        seconds = max(1.5, biggest / GLIDE_DEG_PER_S)
    n = max(1, int(seconds * fps))
    for i in range(1, n + 1):
        a = i / n
        send(rig, {k: start[k] + (target[k] - start[k]) * a for k in target})
        time.sleep(1 / fps)


def play(rig: dict[str, SO101Follower], skill: dict, speed: float = 1.0, should_stop=lambda: False) -> bool:
    """Replay a skill on whichever of its arms are in `rig`. Returns False if interrupted."""
    keys, frames, fps = skill["keys"], skill["frames"], skill["fps"]
    speed = min(speed, 1.0)                       # never faster than it was recorded
    first = dict(zip(keys, frames[0]))
    move_to(rig, first)                           # glide to the start pose first (distance-scaled)
    dt = 1 / (fps * speed)
    t0 = time.perf_counter()
    for i, row in enumerate(frames):
        if should_stop():
            return False
        send(rig, dict(zip(keys, row)))
        # sleep against wall clock so timing doesn't drift
        target_t = t0 + (i + 1) * dt
        time.sleep(max(0.0, target_t - time.perf_counter()))
    return True
