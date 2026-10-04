"""Record one skill by teleoperating with the leader arm(s).

    python record.py handoff                 # all arms in .env ARMS; Enter to start, Enter to stop
    python record.py screwdriver --arms right
    python record.py home --fps 30

Tip: start and end every skill in the SAME "home" pose (every arm parked above the bench).
Then any skill can follow any other skill without a surprise jump.
Two-arm skills record both leaders into one file; a one-arm skill played later leaves
the other arm holding wherever it is.
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
    ap.add_argument("--arms", help=f"comma list, default all: {','.join(arm.ARMS)}")
    args = ap.parse_args()
    arms = arm.parse_arms(args.arms)

    rig = arm.connect_followers(arms)
    leaders = arm.connect_leaders(arms)
    # glide each follower to its leader's current pose first so nothing jumps
    arm.move_to(rig, arm.leader_action(leaders))
    print(f"Teleop live on {', '.join(arms)}. Move the leader(s) to the START pose, then press Enter to begin recording.")
    # mirror leaders -> followers while the user positions the arms
    armed = threading.Event()
    threading.Thread(target=lambda: (input(), armed.set()), daemon=True).start()
    while not armed.is_set():
        arm.send(rig, arm.leader_action(leaders))
        time.sleep(1 / args.fps)

    print(f"RECORDING '{args.name}' … press Enter to stop.")
    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()

    keys, frames = None, []
    dt = 1 / args.fps
    t0 = time.perf_counter()
    i = 0
    while not stop.is_set():
        action = arm.leader_action(leaders)      # all leaders read back to back
        if keys is None:
            keys = list(action.keys())
        frames.append([float(action[k]) for k in keys])
        arm.send(rig, action)
        i += 1
        time.sleep(max(0.0, t0 + i * dt - time.perf_counter()))

    p = arm.save_skill(args.name, keys, frames, args.fps)
    print(f"Saved {len(frames)} frames ({len(frames)/args.fps:.1f}s, arms: {', '.join(arms)}) -> {p}")
    arm.disconnect(rig)
    for lead in leaders.values():
        lead.disconnect()


if __name__ == "__main__":
    sys.exit(main())
