"""Fix motor IDs on an SO-101 arm whose joints answer on the wrong IDs, without unplugging motors.

    python remap_motors.py left_leader          # role from .env (LEFT_LEADER_PORT), or a /dev/tty.usbmodem* port
    python remap_motors.py left_follower

Torque is switched OFF on the whole arm first, so park it on the bench (a follower will go limp).
Then, joint by joint, you move ONLY that joint while it watches which motor ID changes. It prints
the detected mapping and asks before writing anything. IDs are rewritten through temporary IDs
11..16 so no two motors ever share an ID mid-way.

Afterwards the arm MUST be recalibrated (lerobot-calibrate with the same --id as before): the homing
offsets in the calibration file were measured under the old IDs.
"""
import glob
import re
import sys
import threading
import time
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MODEL = "sts3215"
ADDR_ID, ADDR_LOCK, ADDR_POS = (5, 1), (55, 1), (56, 2)
MIN_TRAVEL = 60          # ticks (4096/turn) a joint must move to count; ~5 degrees
TEMP_OFFSET = 10


def resolve_port(arg: str) -> str:
    if arg.startswith("/dev/"):
        return arg
    env = Path(__file__).parent / ".env"
    m = re.search(rf"^{arg.upper()}_PORT=(\S+)", env.read_text(), re.M) if env.exists() else None
    if not m:
        raise SystemExit(f"{arg!r}: not a port and no {arg.upper()}_PORT in .env. Ports: {sorted(glob.glob('/dev/tty.usbmodem*'))}")
    return m.group(1)


def open_bus(port: str) -> FeetechMotorsBus:
    bus = FeetechMotorsBus(port, {j: Motor(i + 1, MODEL, MotorNormMode.RANGE_M100_100) for i, j in enumerate(JOINTS)})
    if not bus.port_handler.openPort():
        raise SystemExit(f"cannot open {port}")
    bus.port_handler.setBaudRate(1_000_000)
    return bus


def read_pos(bus: FeetechMotorsBus, ids: list[int]) -> dict[int, int]:
    out = {}
    for i in ids:
        v, comm, err = bus._read(*ADDR_POS, i)
        if comm == 0 and not err:
            out[i] = v
    return out


def detect(bus: FeetechMotorsBus, ids: list[int], wait=input) -> dict[str, int]:
    """For each joint, watch positions until Enter; return joint -> id that moved most."""
    mapping: dict[str, int] = {}
    for joint in JOINTS:
        print(f"\nMove ONLY the {joint.upper()} joint back and forth a few times, then press Enter.")
        done = threading.Event()
        travel = {i: 0 for i in ids}
        start = read_pos(bus, ids)

        def sample():
            while not done.is_set():
                for i, v in read_pos(bus, ids).items():
                    travel[i] = max(travel[i], abs(v - start.get(i, v)))
                time.sleep(0.05)

        t = threading.Thread(target=sample, daemon=True)
        t.start()
        wait()
        done.set()
        t.join()
        ranked = sorted(travel.items(), key=lambda kv: -kv[1])
        best, second = ranked[0], ranked[1]
        if best[1] < MIN_TRAVEL:
            raise SystemExit(f"  {joint}: nothing moved (max {best[1]} ticks). Aborting, nothing written.")
        note = "" if second[1] < best[1] / 3 else f"   (id {second[0]} also moved {second[1]} ticks, be sure)"
        print(f"  {joint}: id {best[0]} moved {best[1]} ticks{note}")
        mapping[joint] = best[0]
    return mapping


def rewrite_ids(bus: FeetechMotorsBus, mapping: dict[str, int]) -> None:
    targets = {JOINTS.index(j) + 1: cur for j, cur in mapping.items()}       # new_id -> current_id
    for cur in mapping.values():                                             # Lock=0 so EEPROM takes the write
        bus._write(*ADDR_LOCK, cur, 0)
    for cur in mapping.values():                                             # 1) everyone to a temp id
        bus._write(*ADDR_ID, cur, cur + TEMP_OFFSET)
    time.sleep(0.2)
    for new, cur in targets.items():                                         # 2) temp id -> final id
        bus._write(*ADDR_ID, cur + TEMP_OFFSET, new)
    time.sleep(0.2)
    for new in targets:
        bus._write(*ADDR_LOCK, new, 1)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    port = resolve_port(sys.argv[1])
    bus = open_bus(port)
    try:
        ids = sorted((bus.broadcast_ping(num_retry=2) or {}).keys())
        print(f"{port}: motors answering on ids {ids}")
        if ids != list(range(1, 7)):
            raise SystemExit("expected exactly ids 1..6. If some are missing or duplicated use lerobot-setup-motors.")
        temps = [i + TEMP_OFFSET for i in ids]
        if any(bus.ping(i) is not None for i in temps):
            raise SystemExit(f"temporary ids {temps} are already in use on this bus; aborting")
        input("Torque will be switched OFF on all 6 motors: park the arm so it cannot fall, then press Enter.")
        for i in ids:
            bus._disable_torque(i, MODEL)

        mapping = detect(bus, ids)
        wanted = {j: i + 1 for i, j in enumerate(JOINTS)}
        print("\nDetected (joint: current id -> wanted id):")
        for j in JOINTS:
            flag = "" if mapping[j] == wanted[j] else "   <- change"
            print(f"  {j:14s} {mapping[j]} -> {wanted[j]}{flag}")
        if len(set(mapping.values())) != 6:
            raise SystemExit("two joints were detected on the same id; nothing written. Run again and move one joint at a time.")
        if mapping == wanted:
            print("IDs already match the joints. Nothing to write. If teleop is still wrong, check the OTHER arm of the pair.")
            return
        if input("Write these IDs to the motors? [y/N] ").strip().lower() != "y":
            print("nothing written")
            return
        rewrite_ids(bus, mapping)
        after = sorted((bus.broadcast_ping(num_retry=2) or {}).keys())
        print(f"done; ids now answering: {after}")
        if after != list(range(1, 7)):
            raise SystemExit("ID set is not 1..6 after rewrite! Check with `python ports.py` before using the arm.")
        print("Now recalibrate this arm (same --id as before), e.g.\n"
              "  lerobot-calibrate --teleop.type=so101_leader --teleop.port=PORT --teleop.id=leader_so101_left\n"
              "  lerobot-calibrate --robot.type=so101_follower --robot.port=PORT --robot.id=follower_so101_left")
    finally:
        bus.port_handler.closePort()


if __name__ == "__main__":
    main()
