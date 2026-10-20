"""
Motor health check for the SO-101 follower. Writes everything to motor_health.txt.

Usage: python motor_health.py --port COM8 --id my_follower
"""

import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

REGS = [
    "Torque_Enable",
    "Present_Position",
    "Goal_Position",
    "Min_Angle_Limit",
    "Max_Angle_Limit",
    "Present_Voltage",
    "Present_Temperature",
    "Present_Load",
    "Present_Current",
    "Status",
    "Lock",
]

out_lines = []


def log(s: str = "") -> None:
    print(s)
    out_lines.append(s)


def dump(bus, title: str) -> dict:
    log(f"\n=== {title} ===")
    table = {}
    for reg in REGS:
        try:
            table[reg] = bus.sync_read(reg, normalize=False)
        except Exception as e:  # noqa: BLE001
            table[reg] = {m: f"ERR {type(e).__name__}" for m in bus.motors}
    motors = list(bus.motors)
    log(f"{'register':20s}" + "".join(f"{m[:12]:>14s}" for m in motors))
    for reg in REGS:
        log(f"{reg:20s}" + "".join(f"{str(table[reg].get(m, '?')):>14s}" for m in motors))
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", required=True)
    args = ap.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    bus = robot.bus

    log("Calibration file:")
    for m, c in robot.calibration.items():
        log(f"  {m:14s} id={c.id} homing={c.homing_offset:6d} range={c.range_min}..{c.range_max}")

    bus.connect()
    bus.disable_torque()
    dump(bus, "Before torque (raw values)")

    # Safe connect: goal = present, then write calibration and configure (enables torque)
    raw_now = bus.sync_read("Present_Position", normalize=False)
    bus.sync_write("Goal_Position", raw_now, normalize=False)
    bus.write_calibration(robot.calibration)
    robot.configure()
    time.sleep(0.5)
    t = dump(bus, "After torque enabled (arm should NOT have moved)")

    # Nudge test on the three heavy joints, in RAW ticks, toward the calibrated middle.
    log("\n=== Nudge test: 80 raw ticks toward middle of calibrated range, hold 1.5 s ===")
    for m in ["shoulder_lift", "elbow_flex", "wrist_flex", "shoulder_pan"]:
        c = robot.calibration[m]
        mid = (c.range_min + c.range_max) // 2
        start = int(t["Present_Position"][m])
        direction = 1 if mid > start else -1
        goal = start + direction * 80
        bus.sync_write("Goal_Position", {m: goal}, normalize=False)
        trace = []
        for _ in range(15):
            time.sleep(0.1)
            trace.append(int(bus.sync_read("Present_Position", m, normalize=False)[m]))
        load = bus.sync_read("Present_Load", m, normalize=False)[m]
        volt = bus.sync_read("Present_Voltage", m, normalize=False)[m]
        status = bus.sync_read("Status", m, normalize=False)[m]
        moved = trace[-1] - start
        log(
            f"{m:14s} start={start} goal={goal} end={trace[-1]} moved={moved:+d}  "
            f"load={load} volt={volt} status={status}"
        )
        log(f"{'':14s} trace={trace}")
        bus.sync_write("Goal_Position", {m: start}, normalize=False)
        time.sleep(1.0)

    bus.disable_torque()
    bus.disconnect()
    with open("motor_health.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    log("\nSaved motor_health.txt")


if __name__ == "__main__":
    main()
