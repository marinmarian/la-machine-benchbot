"""Thin wrapper around lerobot's SO-101 follower/leader + trajectory file format.

Trajectory file = JSON:
{
  "name": "screwdriver",
  "fps": 30,
  "keys": ["shoulder_pan.pos", ...],
  "frames": [[v0, v1, ...], ...]      # one row per tick, same order as keys
}

Arms are found by their controller board's USB serial (arms.json), so ports and calibration ids
need no editing when arms are swapped or re-plugged. `python arm.py` lists what is connected.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

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

# All four are optional. An id set here must be the arm that is connected (refuses otherwise);
# unset, the one connected follower/leader is used. Ports are only a fallback for a board that
# cannot be identified.
FOLLOWER_PORT = os.environ.get("FOLLOWER_PORT")
LEADER_PORT = os.environ.get("LEADER_PORT")
FOLLOWER_ID = os.environ.get("FOLLOWER_ID")  # calibration file name, e.g. follower_so101
LEADER_ID = os.environ.get("LEADER_ID")
# False (default) keeps the follower holding its pose after a script exits.
DISABLE_TORQUE_ON_DISCONNECT = os.environ.get("DISABLE_TORQUE_ON_DISCONNECT", "false").lower() in ("1", "true", "yes")

SKILLS_DIR = Path(__file__).parent / "skills"
SKILLS_DIR.mkdir(exist_ok=True)


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


# ---------- which arm is on which port ----------

ARMS_FILE = Path(__file__).parent / "arms.json"   # USB serial -> calibration id, learned on first sight
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CALIB_DIRS = {"follower": HF_LEROBOT_CALIBRATION / "robots" / "so_follower",
              "leader": HF_LEROBOT_CALIBRATION / "teleoperators" / "so_leader"}


def _boards() -> dict[str, str]:
    """USB serial -> port for every connected arm controller board (CH343, 1a86:55d3)."""
    from serial.tools import list_ports
    return {p.serial_number: p.device for p in list_ports.comports()
            if p.vid == 0x1A86 and p.pid == 0x55D3 and p.serial_number}


def _role(calib_id: str) -> str | None:
    return next((r for r, d in CALIB_DIRS.items() if (d / f"{calib_id}.json").exists()), None)


def _match_calibration(port: str) -> str | None:
    """Calibration id whose file equals what the motors on `port` have stored (homing offset and
    limits, written there by lerobot-calibrate). Read-only; torque is left as it was."""
    bus = FeetechMotorsBus(port=port, motors={n: Motor(i, "sts3215", MotorNormMode.DEGREES)
                                              for i, n in enumerate(MOTOR_NAMES, 1)})
    try:
        bus.connect()
        stored = {m: (c.homing_offset, c.range_min, c.range_max) for m, c in bus.read_calibration().items()}
    except Exception as e:                    # motors unpowered, or not an SO-101 bus
        print(f"{port}: could not read motors ({e})")
        return None
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
    for d in CALIB_DIRS.values():
        for f in d.glob("*.json"):
            cal = json.loads(f.read_text())
            if all(m in cal and (cal[m]["homing_offset"], cal[m]["range_min"], cal[m]["range_max"]) == stored[m]
                   for m in MOTOR_NAMES):
                return f.stem
    return None


def identify() -> dict[str, str]:
    """calibration id -> port for every connected arm. Known boards come from arms.json; a new
    board is matched once against the calibration files, then remembered."""
    known = json.loads(ARMS_FILE.read_text()) if ARMS_FILE.exists() else {}
    found = {}
    for serial, port in sorted(_boards().items()):
        calib_id = known.get(serial)
        if calib_id is None:
            calib_id = _match_calibration(port)
            if calib_id is None:
                print(f"board {serial} on {port}: motors match no calibration file, not identified")
                continue
            known[serial] = calib_id
            ARMS_FILE.write_text(json.dumps(known, indent=2, sort_keys=True) + "\n")
            print(f"board {serial} on {port}: {calib_id} (new, saved to {ARMS_FILE.name})")
        found[calib_id] = port
    return found


def _resolve(role: str, want_id: str | None, env_port: str | None) -> tuple[str, str]:
    """(calibration id, port) for the follower or leader to connect to."""
    found = identify()
    mine = {i: p for i, p in found.items() if _role(i) == role}
    if want_id:
        if want_id in mine:
            return want_id, mine[want_id]
        if mine:
            raise RuntimeError(f"{role} {want_id!r} is set but the connected {role} is {', '.join(mine)}; "
                               f"plug in {want_id} or change/unset {role.upper()}_ID")
    elif len(mine) == 1:
        return next(iter(mine.items()))
    elif len(mine) > 1:
        raise RuntimeError(f"several {role}s connected ({', '.join(mine)}); set {role.upper()}_ID to pick one")
    # Nothing identified: an unknown board on the configured port (e.g. motors that lost their
    # calibration) still connects the old way, but never a port that belongs to another arm.
    if want_id and env_port and Path(env_port).exists() and env_port not in found.values():
        print(f"{role}: {want_id} not identified, using {role.upper()}_PORT {env_port}")
        return want_id, env_port
    raise RuntimeError(f"no {role} connected (identified: {', '.join(found) or 'none'}); "
                       f"check USB and motor power, then `python arm.py`")


def connect_follower(max_relative_target: float | None = None) -> SO101Follower:
    # max_relative_target caps how far a single command may move a joint, in degrees
    # (lerobot default use_degrees=True; gripper is 0..100). Safety net if a file is corrupt.
    calib_id, port = _resolve("follower", FOLLOWER_ID, FOLLOWER_PORT)
    print(f"follower: {calib_id} on {port}")
    cfg = SO101FollowerConfig(port=port, id=calib_id,
                              max_relative_target=max_relative_target,
                              disable_torque_on_disconnect=DISABLE_TORQUE_ON_DISCONNECT)
    robot = SO101Follower(cfg)
    _connect_with_file_calibration(robot)
    return robot


def connect_leader() -> SO101Leader:
    calib_id, port = _resolve("leader", LEADER_ID, LEADER_PORT)
    print(f"leader: {calib_id} on {port}")
    teleop = SO101Leader(SO101LeaderConfig(port=port, id=calib_id))
    _connect_with_file_calibration(teleop)
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
GRIPPER_MIN = float(os.environ.get("GRIPPER_MIN", "3"))   # never command the gripper below this (0..100);
                                                          # grinding against the closed stop trips overload


def send(robot: SO101Follower, action: dict[str, float]) -> dict[str, float]:
    """send_action with the gripper clamped to >= GRIPPER_MIN. Use this everywhere."""
    if "gripper.pos" in action and action["gripper.pos"] < GRIPPER_MIN:
        action = {**action, "gripper.pos": GRIPPER_MIN}
    return robot.send_action(action)


def disconnect(robot: SO101Follower) -> None:
    """Release the gripper (so it never sits pushing on its stop), then disconnect.
    Arm joints keep holding unless DISABLE_TORQUE_ON_DISCONNECT=true."""
    try:
        robot.bus.disable_torque("gripper")
    finally:
        robot.disconnect()


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
        send(robot, {k: start[k] + (target[k] - start[k]) * a for k in target})
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
        send(robot, dict(zip(keys, row)))
        # sleep against wall clock so timing doesn't drift
        target_t = t0 + (i + 1) * dt
        time.sleep(max(0.0, target_t - time.perf_counter()))
    return True


if __name__ == "__main__":
    arms = identify()
    for calib_id, port in arms.items():
        print(f"{_role(calib_id):8} {calib_id:24} {port}")
    if not arms:
        print("no arms identified")
