"""
Read-only register dump of all six servos. No movement, torque stays off.
Writes regdump.txt.

Usage: python regdump.py --port COM8 --id my_follower
"""

import argparse

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

REGS = [
    "Firmware_Major_Version",
    "Firmware_Minor_Version",
    "Model_Number",
    "Operating_Mode",
    "Phase",
    "Homing_Offset",
    "Min_Position_Limit",
    "Max_Position_Limit",
    "Max_Torque_Limit",
    "Protection_Current",
    "Overload_Torque",
    "Protection_Time",
    "Torque_Enable",
    "Present_Position",
    "Goal_Position",
    "Present_Voltage",
    "Present_Temperature",
    "Status",
    "Lock",
]

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
    motors = list(bus.motors)
    log(f"{'register':24s}" + "".join(f"{m[:12]:>14s}" for m in motors))
    for reg in REGS:
        row = {}
        for m in motors:
            try:
                row[m] = bus.read(reg, m, normalize=False)
            except Exception as e:  # noqa: BLE001
                row[m] = f"ERR:{type(e).__name__}"
        log(f"{reg:24s}" + "".join(f"{str(row[m]):>14s}" for m in motors))
    bus.disconnect()
    with open("regdump.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    log("\nSaved regdump.txt")


if __name__ == "__main__":
    main()
