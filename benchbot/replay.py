"""Replay a recorded skill (no voice). Good for testing each recording.

    python replay.py screwdriver
    python replay.py handoff --speed 0.7
    python replay.py handoff --arms right     # only drive the right arm's part
    python replay.py --list
"""
import argparse

import arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--arms", help="comma list; default: the arms the skill uses")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.name:
        for s in arm.list_skills():
            print(f"{s:20s} {', '.join(arm.skill_arms(arm.load_skill(s)))}")
        if not arm.list_skills():
            print("(no skills recorded yet)")
        return

    skill = arm.load_skill(args.name)
    arms = arm.parse_arms(args.arms) if args.arms else arm.skill_arms(skill)
    rig = arm.connect_followers(arms)
    try:
        arm.play(rig, skill, speed=args.speed)
    finally:
        arm.disconnect(rig)


if __name__ == "__main__":
    main()
