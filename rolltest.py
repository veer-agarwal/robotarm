"""
Controlled experiments on ONE servo (default wrist_roll, which has no load) to find
why it drives away from its own goal. Each trial: write a setting, set goal = present,
enable torque, watch for 0.8 s, abort if it moves more than 150 ticks, torque off.
Original settings are restored at the end. Writes rolltest.txt.

Usage: python rolltest.py --port COM8 --id my_follower [--motor wrist_roll]
"""

import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

out = []


def log(s: str = "") -> None:
    print(s)
    out.append(s)


def trial(bus, m, label):
    bus.disable_torque(m)
    time.sleep(0.2)
    start = int(bus.read("Present_Position", m, normalize=False))
    bus.write("Goal_Position", m, start, normalize=False)
    bus.enable_torque(m)
    trace = []
    aborted = False
    for _ in range(8):
        time.sleep(0.1)
        p = int(bus.read("Present_Position", m, normalize=False))
        trace.append(p)
        if abs(p - start) > 150:
            aborted = True
            break
    hold_ok = not aborted and abs(trace[-1] - start) < 15
    result = f"{label:40s} start={start} trace={trace} -> {'HOLDS' if hold_ok else 'RUNS AWAY'}"
    step_ok = None
    if hold_ok:
        # try a +40 step
        bus.write("Goal_Position", m, start + 40, normalize=False)
        time.sleep(0.6)
        p2 = int(bus.read("Present_Position", m, normalize=False))
        bus.write("Goal_Position", m, start, normalize=False)
        time.sleep(0.6)
        step_ok = abs((p2 - start) - 40) < 12
        result += f"; +40 step moved {p2 - start:+d} -> {'OK' if step_ok else 'WRONG'}"
    bus.disable_torque(m)
    log(result)
    return hold_ok and bool(step_ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--motor", default="wrist_roll")
    args = ap.parse_args()
    m = args.motor

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    bus = robot.bus
    bus.connect()
    bus.disable_torque()
    bus.write("Lock", m, 0, normalize=False)

    orig_offset = int(bus.read("Homing_Offset", m, normalize=False))
    orig_phase = int(bus.read("Phase", m, normalize=False))
    orig_min = int(bus.read("Min_Position_Limit", m, normalize=False))
    orig_max = int(bus.read("Max_Position_Limit", m, normalize=False))
    log(f"Motor {m}: original Homing_Offset={orig_offset} Phase={orig_phase} limits={orig_min}..{orig_max}")
    bus.write("Min_Position_Limit", m, 0, normalize=False)
    bus.write("Max_Position_Limit", m, 4095, normalize=False)

    try:
        log("\n--- Trial A: as-is ---")
        if trial(bus, m, "as-is"):
            log("Already fine?!")

        log("\n--- Trial B: Homing_Offset = 0 ---")
        bus.write("Homing_Offset", m, 0, normalize=False)
        time.sleep(0.2)
        b = trial(bus, m, "offset=0")
        bus.write("Homing_Offset", m, orig_offset, normalize=False)
        time.sleep(0.2)
        if b:
            log("RESULT: the homing offset is what breaks it.")
            return

        log("\n--- Trial C: Phase variants (offset restored) ---")
        for ph in [orig_phase ^ 0x01, orig_phase ^ 0x02, orig_phase ^ 0x04, orig_phase ^ 0x08, orig_phase ^ 0x10]:
            bus.write("Phase", m, ph, normalize=False)
            time.sleep(0.2)
            got = int(bus.read("Phase", m, normalize=False))
            ok = trial(bus, m, f"phase={ph} (readback {got})")
            if ok:
                log(f"RESULT: Phase={ph} makes this motor behave.")
                break
    finally:
        bus.disable_torque(m)
        bus.write("Phase", m, orig_phase, normalize=False)
        bus.write("Homing_Offset", m, orig_offset, normalize=False)
        bus.write("Min_Position_Limit", m, orig_min, normalize=False)
        bus.write("Max_Position_Limit", m, orig_max, normalize=False)
        log(f"\nRestored Phase={orig_phase} Homing_Offset={orig_offset} limits={orig_min}..{orig_max}")
        bus.disconnect()
        with open("rolltest.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(out))
        log("Saved rolltest.txt")


if __name__ == "__main__":
    main()
