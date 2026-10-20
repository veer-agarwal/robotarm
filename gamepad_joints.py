"""
Joint-space Xbox control for a single SO-101 follower arm.

No inverse kinematics, so it runs on Windows without the placo package.
Each stick drives a joint directly. Every tick the goal is "where the joint
is now, plus a small lead in the stick direction", so it can never wind up
against an end stop.

    Left stick  X : shoulder_pan        Left stick  Y : shoulder_lift
    Right stick X : wrist_flex          Right stick Y : elbow_flex
    LB / RB       : wrist_roll          LT / RT       : gripper close / open
    B             : exit (arm goes limp, so hold it)

Usage:
    python gamepad_joints.py --port COM8 --id my_follower
"""

import argparse
import time

import pygame

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

FPS = 30
DEADZONE = 0.15

# Degrees of lead at full stick deflection. Bigger = faster. Start small.
LEAD = {
    "shoulder_pan": 4.0,
    "shoulder_lift": 4.0,
    "elbow_flex": 4.0,
    "wrist_flex": 4.0,
    "wrist_roll": 10.0,
    "gripper": 6.0,
}

# Flip a sign here if a joint moves the opposite way to what feels natural.
SIGN = {
    "shoulder_pan": 1,
    "shoulder_lift": -1,
    "elbow_flex": -1,
    "wrist_flex": 1,
    "wrist_roll": 1,
    "gripper": 1,
}

# pygame / SDL2 mapping for an Xbox controller on Windows (confirmed by diag.py)
AX_LX, AX_LY, AX_RX, AX_RY, AX_LT, AX_RT = 0, 1, 2, 3, 4, 5
BTN_B, BTN_LB, BTN_RB = 1, 4, 5


def dz(v: float) -> float:
    return 0.0 if abs(v) < DEADZONE else v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", required=True)
    args = ap.parse_args()

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise SystemExit("No gamepad found. Plug it in and try again.")
    pad = pygame.joystick.Joystick(0)
    pad.init()
    print(f"Gamepad: {pad.get_name()}")

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.id,
            use_degrees=True,
            max_relative_target=12.0,  # safety cap per tick
            position_p_coefficient=32,  # stiffer hold against gravity (default 16)
        )
    )
    # Safe connect: tell every servo "your goal is where you are now" BEFORE
    # enabling torque, so the arm holds still instead of lunging to a stale goal.
    robot.bus.connect()
    robot.bus.disable_torque()

    # Turn-counter check: a servo whose internal position is a full turn away from
    # what it reports will run away the moment torque comes on. Refuse to start.
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
        raise SystemExit(
            f"These servos are a full turn out of step: {bad}. "
            "Unplug the 12 V supply for 10 s, plug it back in, and run again."
        )

    raw_now = robot.bus.sync_read("Present_Position", normalize=False)
    robot.bus.sync_write("Goal_Position", raw_now, normalize=False)
    robot.bus.write_calibration(robot.calibration)
    robot.configure()  # sets PID and enables torque; goal already equals current pose
    print("Connected. Press B to quit.\n")

    obs0 = robot.get_observation()
    hold = {k.removesuffix(".pos"): v for k, v in obs0.items() if k.endswith(".pos")}

    try:
        while True:
            t0 = time.perf_counter()
            pygame.event.pump()
            if pad.get_button(BTN_B):
                break

            lx, ly = dz(pad.get_axis(AX_LX)), dz(pad.get_axis(AX_LY))
            rx, ry = dz(pad.get_axis(AX_RX)), dz(pad.get_axis(AX_RY))
            lt = (pad.get_axis(AX_LT) + 1) / 2  # triggers rest at -1
            rt = (pad.get_axis(AX_RT) + 1) / 2
            roll = float(pad.get_button(BTN_RB)) - float(pad.get_button(BTN_LB))

            stick = {
                "shoulder_pan": lx,
                "shoulder_lift": ly,
                "elbow_flex": ry,
                "wrist_flex": rx,
                "wrist_roll": roll,
                "gripper": rt - lt,
            }

            obs = robot.get_observation()
            now = {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}

            # While a stick is deflected, the goal leads the current position.
            # When it is centred, keep the last goal so gravity can't creep the joint down.
            for j, p in now.items():
                if stick[j] != 0.0:
                    hold[j] = p + SIGN[j] * stick[j] * LEAD[j]
            robot.send_action({f"{j}.pos": v for j, v in hold.items()})

            line = "  ".join(f"{j[:5]}:{now[j]:6.1f}" for j in now)
            sticks = f"LX{lx:+.1f} LY{ly:+.1f} RX{rx:+.1f} RY{ry:+.1f}"
            print(f"{sticks} | {line}   ", end="\r", flush=True)

            time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))
    finally:
        print("\nDisconnecting. Hold the arm, it will go limp.")
        robot.disconnect()
        pygame.quit()


if __name__ == "__main__":
    main()
