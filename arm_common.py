"""Shared helpers: safe connect (no lunge, turn-counter check) and joint dict extraction."""

import time


def joints(obs: dict) -> dict:
    return {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}


def safe_connect(robot, acceleration: int = 80) -> None:
    """Connect so the arm holds still: goal = present before torque on. Refuses to start if a
    servo's internal turn counter is out of step (it would run away)."""
    robot.bus.connect()
    robot.bus.disable_torque()
    bad = []
    for m in robot.bus.motors:
        robot.bus.write("Lock", m, 0, normalize=False)
        phase = int(robot.bus.read("Phase", m, normalize=False))
        wrapped = int(robot.bus.read("Present_Position", m, normalize=False))
        robot.bus.write("Phase", m, phase | 0x10, normalize=False)
        time.sleep(0.03)
        internal = int(robot.bus.read("Present_Position", m, normalize=False))
        robot.bus.write("Phase", m, phase, normalize=False)
        time.sleep(0.03)
        if abs(internal - wrapped) > 2048:
            bad.append(m)
    if bad:
        robot.bus.disconnect()
        raise SystemExit(f"Servos a full turn out of step: {bad}. Unplug the 12 V supply for 10 s and rerun.")
    raw_now = robot.bus.sync_read("Present_Position", normalize=False)
    robot.bus.sync_write("Goal_Position", raw_now, normalize=False)
    robot.bus.write_calibration(robot.calibration)
    for cam in getattr(robot, "cameras", {}).values():
        cam.connect()
    robot.configure()
    for m in robot.bus.motors:
        robot.bus.write("Acceleration", m, acceleration, normalize=False)
