"""Thin wrapper around lerobot's SO-101 follower/leader + trajectory file format.

Several arms can be driven together. Each arm has a name (default: "right", "left");
every joint key is prefixed with it, so a two-arm skill looks like:

{
  "name": "handoff",
  "fps": 30,
  "keys": ["right_shoulder_pan.pos", ..., "left_shoulder_pan.pos", ...],
  "frames": [[v0, v1, ...], ...],     # one row per tick, same order as keys
  "hover_frame": 159                  # optional: where the wrist camera lines up on the tool
}

Older single-arm files have unprefixed keys ("shoulder_pan.pos"); load_skill() maps those
to the FIRST arm listed in ARMS, so they keep playing unchanged.

A "rig" is just a dict {arm_name: SO101Follower}; leaders are {arm_name: SO101Leader}.

Each arm's boards are found by their USB serial (arms.json), so ports need no editing when arms
are re-plugged; .env only names each arm's calibration ids. `python arm.py` lists what is connected.
"""
from __future__ import annotations

import functools
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


def parse_arms(spec: str | None) -> list[str]:
    """'left,right' -> ['left', 'right']; None/'' -> all configured arms. Validates names."""
    arms = [a.strip() for a in spec.split(",") if a.strip()] if spec else list(ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; ARMS in .env = {ARMS}")
    return arms


# ---------- which board is which arm ----------

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


@functools.cache
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


def _resolve(arm: str, role: str) -> tuple[str, str]:
    """(calibration id, port) of `arm`'s follower or leader. {ARM}_{ROLE}_ID in .env pins the
    calibration id (refuses if another arm of that role is plugged in instead); unset, the one
    connected board of that role is used. {ARM}_{ROLE}_PORT is only a fallback for a board
    that can't be identified."""
    var = f"{arm.upper()}_{role.upper()}"
    want_id, env_port = os.environ.get(f"{var}_ID"), os.environ.get(f"{var}_PORT")
    found = identify()
    mine = {i: p for i, p in found.items() if _role(i) == role}
    if want_id:
        if want_id in mine:
            return want_id, mine[want_id]
        if mine:
            raise RuntimeError(f"{var}_ID is {want_id!r} but the connected {role}s are {', '.join(mine)}; "
                               f"plug in {want_id} or change/unset {var}_ID")
    elif len(mine) == 1:
        return next(iter(mine.items()))
    elif len(mine) > 1:
        raise RuntimeError(f"several {role}s connected ({', '.join(mine)}); set {var}_ID to pick one")
    # Nothing identified: an unknown board on the configured port (e.g. motors that lost their
    # calibration) still connects the old way, but never a port that belongs to another arm.
    taken = {p.replace("/dev/tty.", "/dev/cu.") for p in found.values()}     # macOS lists both names
    if want_id and env_port and Path(env_port).exists() and env_port.replace("/dev/tty.", "/dev/cu.") not in taken:
        print(f"{arm} {role}: {want_id} not identified, using {var}_PORT {env_port}")
        return want_id, env_port
    raise RuntimeError(f"no {role} connected (identified: {', '.join(found) or 'none'}); "
                       f"check USB and motor power, then `python arm.py`")


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
    calib_id, port = _resolve(arm, "follower")
    print(f"{arm} follower: {calib_id} on {port}")
    cfg = SO101FollowerConfig(port=port, id=calib_id,
                              max_relative_target=max_relative_target,
                              disable_torque_on_disconnect=DISABLE_TORQUE_ON_DISCONNECT)
    robot = SO101Follower(cfg)
    _connect_with_file_calibration(robot)
    return robot


def connect_leader(arm: str) -> SO101Leader:
    calib_id, port = _resolve(arm, "leader")
    print(f"{arm} leader: {calib_id} on {port}")
    teleop = SO101Leader(SO101LeaderConfig(port=port, id=calib_id))
    _connect_with_file_calibration(teleop)
    return teleop


def _check_distinct(arm: str, role: str, up: dict) -> None:
    calib_id = _resolve(arm, role)[0]
    for other, dev in up.items():
        if dev.id == calib_id:
            raise RuntimeError(f"arms {other} and {arm} both resolve to {calib_id}; set "
                               f"{other.upper()}_{role.upper()}_ID and {arm.upper()}_{role.upper()}_ID in .env")


def connect_followers(arms: list[str] | None = None) -> dict[str, SO101Follower]:
    """Connect the followers of the given arms (default: all). On failure, disconnect
    the ones already up so no arm is left half-configured."""
    rig: dict[str, SO101Follower] = {}
    try:
        for a in arms or ARMS:
            try:
                _check_distinct(a, "follower", rig)
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
            _check_distinct(a, "leader", leaders)
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


def update_skill(name: str, skill: dict) -> Path:
    """Write a loaded skill back, keeping the file's own key spelling (legacy files stay unprefixed)."""
    p = skill_path(name)
    p.write_text(json.dumps({**skill, "keys": json.loads(p.read_text())["keys"]}))
    return p


def solo(skill: dict, arm: str) -> dict:
    """One arm's part of a loaded skill, with plain keys ("shoulder_pan.pos"), for the one-arm tools
    (adjust, wristrec, visual_servoing). on(arm, ...) makes it playable again."""
    cols = [i for i, k in enumerate(skill["keys"]) if k.startswith(arm + "_")]
    if not cols:
        raise SystemExit(f"skill {skill.get('name')!r} has no {arm} arm (it uses {', '.join(skill_arms(skill))})")
    return {**skill, "keys": [skill["keys"][i][len(arm) + 1:] for i in cols],
            "frames": [[row[i] for i in cols] for row in skill["frames"]]}


def on(arm: str, skill: dict) -> dict:
    """A plain-key (solo) skill, prefixed for `arm` so play() can drive it."""
    return {**skill, "keys": [f"{arm}_{k}" for k in skill["keys"]]}


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


if __name__ == "__main__":
    names = {os.environ.get(f"{a.upper()}_{r}_ID"): a for a in ARMS for r in ("FOLLOWER", "LEADER")}
    arms = identify()
    for calib_id, port in arms.items():
        print(f"{names.get(calib_id, '-'):6} {_role(calib_id):8} {calib_id:24} {port}")
    if not arms:
        print("no arms identified")
