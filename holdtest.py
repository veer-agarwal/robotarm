"""
Hold test for all six servos, one at a time: goal = present, torque on 0.8 s,
abort instantly if it moves more than 150 ticks. Also reports the servo's internal
(unwrapped) position by briefly reading with Phase bit 4 set, so we can see whether
the servo thinks it is a full turn away from where LeRobot thinks it is.
Writes holdtest.txt. Settings are restored.

Usage: python holdtest.py --port COM8 --id my_follower
"""

import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

out = []


def log(s: str = "") -> None:
    print(s)
    out.append(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", required=True)
    args = ap.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    bus = robot.bus
    bus.connect()
    bus.disable_torque()

    for m in bus.motors:
        bus.write("Lock", m, 0, normalize=False)
        phase = int(bus.read("Phase", m, normalize=False))
        wrapped = int(bus.read("Present_Position", m, normalize=False))
        # peek at the unwrapped internal position
        bus.write("Phase", m, phase | 0x10, normalize=False)
        time.sleep(0.05)
        internal = int(bus.read("Present_Position", m, normalize=False))
        bus.write("Phase", m, phase, normalize=False)
        time.sleep(0.05)
        turns = round((internal - wrapped) / 4096)
        log(f"{m:14s} reported={wrapped:5d} internal={internal:6d} -> {turns:+d} turn(s) off")

    log("\nHold test:")
    for m in bus.motors:
        bus.disable_torque(m)
        time.sleep(0.1)
        start = int(bus.read("Present_Position", m, normalize=False))
        bus.write("Goal_Position", m, start, normalize=False)
        bus.enable_torque(m)
        trace = []
        for _ in range(8):
            time.sleep(0.1)
            p = int(bus.read("Present_Position", m, normalize=False))
            trace.append(p)
            if abs(p - start) > 150:
                break
        bus.disable_torque(m)
        ok = abs(trace[-1] - start) < 15
        log(f"{m:14s} start={start} trace={trace} -> {'HOLDS' if ok else 'RUNS AWAY'}")

    bus.disconnect()
    with open("holdtest.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    log("\nSaved holdtest.txt")


if __name__ == "__main__":
    main()
