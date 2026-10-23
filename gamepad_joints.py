"""
Drive the SO-101 with joint-space Xbox control and a live wrist camera view.

    Left stick  X : shoulder_pan        Left stick  Y : shoulder_lift
    Right stick X : wrist_flex          Right stick Y : elbow_flex
    LB / RB       : wrist_roll          LT / RT       : gripper close / open
    L3 (click left stick) : return to start pose     B : quit (hold the arm, it goes limp)

Usage: python gamepad_joints.py --port COM8 --id my_follower [--cam 0]
Tuning (speed, directions, strength) lives at the top of joint_control.py.
"""

import argparse
import time

import cv2
import pygame

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from arm_common import joints, safe_connect
from joint_control import JointTeleop

FPS = 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--cam", type=int, default=0, help="wrist camera index, -1 to disable")
    args = ap.parse_args()

    pygame.init(); pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise SystemExit("No gamepad found.")
    pad = pygame.joystick.Joystick(0); pad.init()

    cap = None
    if args.cam >= 0:
        cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            cap = None

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True, max_relative_target=12.0))
    safe_connect(robot)
    tele = JointTeleop(pad, joints(robot.get_observation()))
    print("Connected. B quits.\n")

    try:
        while True:
            t0 = time.perf_counter()
            inp = tele.read()
            if inp["B"]:
                break
            now = joints(robot.get_observation())
            robot.send_action(tele.step(now, inp))
            if cap is not None:
                ok, frame = cap.read()
                if ok:
                    cv2.imshow("wrist cam", frame); cv2.waitKey(1)
            print(tele.status() + "   ", end="\r", flush=True)
            time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))
    finally:
        print("\nDisconnecting. Hold the arm, it will go limp.")
        robot.disconnect()
        if cap is not None:
            cap.release(); cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    main()
