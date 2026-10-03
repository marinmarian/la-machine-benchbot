"""Replay a recorded skill (no voice). Good for testing each recording.

    python replay.py screwdriver
    python replay.py screwdriver --speed 0.7
    python replay.py --list
"""
import argparse

import arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.name:
        print("\n".join(arm.list_skills()) or "(no skills recorded yet)")
        return

    robot = arm.connect_follower()
    try:
        arm.play(robot, arm.load_skill(args.name), speed=args.speed)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
