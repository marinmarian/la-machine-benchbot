"""Which SO-101 arm is on which serial port? Read-only.

    python ports.py
    python ports.py --watch        # 8 s: wiggle ONE arm by hand, it prints which port moved

Pings motor ids 1..6 on every /dev/tty.usbmodem* and prints a leader/follower guess:
a follower that lerobot has configured before has torque on and P_Coefficient 16; a
leader has torque off and P 32. On a never-configured pair both read P 32, so go by
torque / which one has the handle and confirm by unplugging one USB cable at a time.
"No motors answer" almost always means the arm's power supply is off (the USB board
enumerates without it) or the 3-pin bus cable to the first motor is loose.
"""
import glob
import sys
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def probe(port: str) -> None:
    motors = {n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(NAMES)}
    bus = FeetechMotorsBus(port, motors)
    try:
        bus.port_handler.openPort()
        bus.port_handler.setBaudRate(1_000_000)
        ids = sorted((bus.broadcast_ping(num_retry=2) or {}).keys())
        if not ids:
            print(f"{port}: no motors answer (power off? bus cable?)")
            return
        torque = p_coef = 0
        for n, m in motors.items():
            if m.id in ids:
                torque += bus.read("Torque_Enable", n, normalize=False)
                p_coef += bus.read("P_Coefficient", n, normalize=False)
        guess = "follower" if torque else ("follower (configured, torque off)" if p_coef / len(ids) < 32 else "leader?")
        extra = "" if ids == list(range(1, 7)) else "  <- not ids 1..6: run lerobot-setup-motors"
        print(f"{port}: ids {ids}  torque_on={torque}/{len(ids)}  P~{p_coef / len(ids):.0f}  -> {guess}{extra}")
    except Exception as e:  # noqa: BLE001
        print(f"{port}: {type(e).__name__}: {e}")
    finally:
        try:
            bus.port_handler.closePort()
        except Exception:  # noqa: BLE001
            pass


def watch(ports: list[str], seconds: float = 8.0) -> None:
    """Sample raw positions on every port and report which arm was moved by hand."""
    buses = {}
    for port in ports:
        motors = {n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(NAMES)}
        bus = FeetechMotorsBus(port, motors)
        bus.port_handler.openPort()
        bus.port_handler.setBaudRate(1_000_000)
        buses[port] = bus

    def positions(bus):
        return {n: bus.read("Present_Position", n, normalize=False) for n in NAMES}

    start = {port: positions(bus) for port, bus in buses.items()}
    moved = {port: {n: 0 for n in NAMES} for port in buses}
    print(f"Watching for {seconds:.0f}s: move ONE arm by hand now…")
    t0 = time.time()
    while time.time() - t0 < seconds:
        for port, bus in buses.items():
            for n, v in positions(bus).items():
                moved[port][n] = max(moved[port][n], abs(v - start[port][n]))
        time.sleep(0.1)
    for port, bus in buses.items():
        bus.port_handler.closePort()
        big = {n: d for n, d in moved[port].items() if d > 40}      # 40 ticks ~ 3.5 degrees
        print(f"{port}: {'MOVED ' + ', '.join(big) if big else 'still'}")


if __name__ == "__main__":
    ports = sorted(glob.glob("/dev/tty.usbmodem*"))
    if not ports:
        print("no /dev/tty.usbmodem* ports")
    elif "--watch" in sys.argv:
        watch(ports)
    else:
        for p in ports:
            probe(p)
