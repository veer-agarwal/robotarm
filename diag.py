"""
Diagnostic for SO-101 + Xbox controller.
1) Prints live gamepad axes for 10 s: move each stick and note which axis number changes.
2) Nudges every joint 5 degrees and back, reporting whether it moved.

Usage: python diag.py --port COM8 --id my_follower
"""

import argparse
import time

import pygame

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", required=True)
    args = ap.parse_args()

    # ---- Part 1: gamepad axes ----
    pygame.init()
    pygame.joystick.init()
    pad = pygame.joystick.Joystick(0)
    pad.init()
    n = pad.get_numaxes()
    print(f"\nGamepad: {pad.get_name()} with {n} axes, {pad.get_numbuttons()} buttons")
    print("Move LEFT stick, then RIGHT stick, then pull triggers. Watching for 10 s...\n")
    t_end = time.time() + 10
    while time.time() < t_end:
        pygame.event.pump()
        vals = "  ".join(f"ax{i}:{pad.get_axis(i):+.2f}" for i in range(n))
        btns = "".join(str(i) for i in range(pad.get_numbuttons()) if pad.get_button(i))
        print(f"{vals}   buttons:{btns:<8}", end="\r", flush=True)
        time.sleep(0.05)
    print("\n")
    pygame.quit()

    # ---- Part 2: motor nudge ----
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    robot.connect()
    obs = robot.get_observation()
    pos = {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}
    print("Current joint positions (deg):")
    for k, v in pos.items():
        print(f"  {k:14s} {v:7.1f}")
    print("\nNudging each joint +5 deg then back. Keep hands clear.\n")
    for joint, start in pos.items():
        robot.send_action({f"{joint}.pos": start + 5})
        time.sleep(0.6)
        after = robot.get_observation()[f"{joint}.pos"]
        robot.send_action({f"{joint}.pos": start})
        time.sleep(0.6)
        moved = abs(after - start)
        verdict = "MOVED" if moved > 2 else "DID NOT MOVE"
        print(f"  {joint:14s} commanded +5, measured {moved:+.1f}  -> {verdict}")
    robot.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
