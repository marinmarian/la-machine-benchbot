"""Record one skill by teleoperating with the leader arm.

    python record.py screwdriver            # press Enter to start, Enter again to stop
    python record.py home --fps 30

Tip: start and end every skill in the SAME "home" pose (arm parked above the bench).
Then any skill can follow any other skill without a surprise jump.
"""
import argparse
import sys
import threading
import time

import arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    robot = arm.connect_follower()
    leader = arm.connect_leader()
    # glide the follower to the leader's current pose first so it never jumps
    arm.move_to(robot, leader.get_action(), seconds=2.0)
    print("Teleop live. Move the leader to the START pose, then press Enter to begin recording.")
    # mirror leader -> follower while the user positions the arm
    armed = threading.Event()
    threading.Thread(target=lambda: (input(), armed.set()), daemon=True).start()
    while not armed.is_set():
        robot.send_action(leader.get_action())
        time.sleep(1 / args.fps)

    print(f"RECORDING '{args.name}' … press Enter to stop.")
    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()

    keys, frames = None, []
    dt = 1 / args.fps
    t0 = time.perf_counter()
    i = 0
    while not stop.is_set():
        action = leader.get_action()
        if keys is None:
            keys = list(action.keys())
        frames.append([float(action[k]) for k in keys])
        robot.send_action(action)
        i += 1
        time.sleep(max(0.0, t0 + i * dt - time.perf_counter()))

    p = arm.save_skill(args.name, keys, frames, args.fps)
    print(f"Saved {len(frames)} frames ({len(frames)/args.fps:.1f}s) -> {p}")
    robot.disconnect()
    leader.disconnect()


if __name__ == "__main__":
    sys.exit(main())
