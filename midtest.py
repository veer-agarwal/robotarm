"""
Mid-range servo test. Torque off -> you pose the arm by hand -> torque on at that pose
-> each heavy joint is asked to move +40 ticks, then -40 ticks, while we log position and load.
Writes midtest.txt.

Usage: python midtest.py --port COM8 --id my_follower
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
    # Open the angle limits wide for this test so they can't interfere
    bus.sync_write("Lock", 0, normalize=False)
    bus.sync_write("Min_Angle_Limit", 0, normalize=False)
    bus.sync_write("Max_Angle_Limit", 4095, normalize=False)

    print("\nMotors are LIMP. By hand, pose the arm: upper arm vertical, forearm horizontal,")
    print("gripper pointing forward, base facing straight ahead. Hold it there and press Enter.")
    input()

    raw = bus.sync_read("Present_Position", normalize=False)
    bus.sync_write("Goal_Position", raw, normalize=False)
    bus.enable_torque()
    log(f"Torque on at raw positions: {raw}")
    print("You can let go now. Watching for 2 s to see if it holds...")
    time.sleep(2.0)
    held = bus.sync_read("Present_Position", normalize=False)
    for m in raw:
        log(f"  {m:14s} start={raw[m]} after2s={held[m]} drift={held[m]-raw[m]:+d}")

    for m in ["shoulder_lift", "elbow_flex", "wrist_flex", "shoulder_pan"]:
        for step in (+40, -40):
            start = int(bus.sync_read("Present_Position", m, normalize=False)[m])
            goal = start + step
            bus.sync_write("Goal_Position", {m: goal}, normalize=False)
            trace, loads = [], []
            for _ in range(12):
                time.sleep(0.1)
                trace.append(int(bus.sync_read("Present_Position", m, normalize=False)[m]))
                loads.append(int(bus.sync_read("Present_Load", m, normalize=False)[m]))
            moved = trace[-1] - start
            ok = "OK" if abs(moved - step) < 12 else ("RUNAWAY/REVERSED" if moved * step < 0 else "WEAK/STUCK")
            log(f"{m:14s} step={step:+d} start={start} end={trace[-1]} moved={moved:+d} -> {ok}")
            log(f"{'':14s} pos ={trace}")
            log(f"{'':14s} load={loads}")
            time.sleep(0.5)

    bus.disable_torque()
    bus.disconnect()
    with open("midtest.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    log("\nSaved midtest.txt. Motors are limp again; support the arm.")


if __name__ == "__main__":
    main()
