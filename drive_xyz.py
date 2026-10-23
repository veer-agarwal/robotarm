"""
Drive the SO-101 with Cartesian (move-the-gripper) gamepad control and a live wrist camera view.

    Left stick : forward/back, left/right     Right stick up/down : up/down
    D-pad up/down : tilt gripper              LB/RB : wrist roll      LT/RT : gripper
    L3 (click left stick) : return to start pose     B : quit (hold the arm, it goes limp)

Usage: python drive_xyz.py --port COM8 --id my_follower [--cam 0]
"""

import argparse
import time

import cv2
import pygame

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from arm_common import joints, safe_connect
from xyz_control import CartesianTeleop

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

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True, max_relative_target=12.0, position_p_coefficient=20))
    safe_connect(robot)
    now = joints(robot.get_observation())
    tele = CartesianTeleop(pad, now, robot.calibration)
    print("Connected. Move the gripper with the sticks. B quits.\n")
    log = open("xyz_log.csv", "w")
    log.write("t,fwd,left,up,tx,ty,tz,pitch," + ",".join(f"q_{j}" for j in tele.q) + ","
              + ",".join(f"now_{j}" for j in now) + "," + ",".join(f"goal_{j}" for j in now) + "\n")
    tstart = time.perf_counter()

    try:
        while True:
            t0 = time.perf_counter()
            inp = tele.read()
            if inp["B"]:
                break
            now = joints(robot.get_observation())
            goal = tele.step(now, inp)
            robot.send_action(goal)
            log.write(f"{t0-tstart:.3f},{inp['fwd']:.2f},{inp['left']:.2f},{inp['up']:.2f},"
                      f"{tele.target[0]:.4f},{tele.target[1]:.4f},{tele.target[2]:.4f},{tele.pitch:.1f},"
                      + ",".join(f"{v:.1f}" for v in tele.q.values()) + ","
                      + ",".join(f"{v:.1f}" for v in now.values()) + ","
                      + ",".join(f"{goal[j + '.pos']:.1f}" for j in now) + "\n")
            if cap is not None:
                ok, frame = cap.read()
                if ok:
                    cv2.imshow("wrist cam", frame); cv2.waitKey(1)
            print(tele.status() + "   ", end="\r", flush=True)
            time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))
    finally:
        print("\nDisconnecting. Hold the arm, it will go limp.")
        log.close()
        robot.disconnect()
        if cap is not None:
            cap.release(); cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    main()
