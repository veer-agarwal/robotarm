"""
Record demonstrations for the SO-101 follower with Xbox joint control and the wrist camera,
saved as a LeRobot dataset ready for `lerobot-train`.

Controls while recording (same as gamepad_joints.py):
    Left stick  X : shoulder_pan        Left stick  Y : shoulder_lift
    Right stick X : wrist_flex          Right stick Y : elbow_flex
    LB / RB       : wrist_roll          LT / RT       : gripper close / open
    L3 (click left stick) : glide back to the start pose
    Y : episode SUCCEEDED -> save it         X : scrap this episode and redo it
    A : start the next episode (after a reset pause)
    B : stop recording and finalize the dataset

Usage:
    python record_gamepad.py --port COM8 --id my_follower --repo veer/so101_pick_block ^
        --task "Pick up the block" --episodes 50
    add --resume to append to an existing dataset.
"""

import argparse
import time

import cv2
import pygame

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

from arm_common import joints, safe_connect
from joint_control import JointTeleop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--repo", required=True, help="dataset name, e.g. veer/so101_pick_block")
    ap.add_argument("--task", required=True, help="one sentence describing the task")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    pygame.init(); pygame.joystick.init()
    pad = pygame.joystick.Joystick(0); pad.init()

    robot = SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.id, use_degrees=True, max_relative_target=12.0,
        cameras={"wrist": OpenCVCameraConfig(index_or_path=args.cam, fps=args.fps, width=640, height=480)},
    ))
    safe_connect(robot)

    features = {
        **hw_to_dataset_features(robot.action_features, "action", use_video=True),
        **hw_to_dataset_features(robot.observation_features, "observation", use_video=True),
    }
    if args.resume:
        from lerobot.utils.constants import HF_LEROBOT_HOME
        dataset = LeRobotDataset.resume(args.repo, root=HF_LEROBOT_HOME / args.repo, image_writer_threads=4)
        done = dataset.num_episodes
    else:
        dataset = LeRobotDataset.create(args.repo, args.fps, features, robot_type=robot.name,
                                        use_videos=True, image_writer_threads=4)
        done = 0
    print(f"Dataset at {dataset.root}  ({done} episodes already)")

    tele = JointTeleop(pad, joints(robot.get_observation()))
    ep = done
    recording = False
    frames = 0
    print("\nReady. Press A to start episode 1. Y = success, X = redo, B = quit.\n")

    try:
        while ep < args.episodes:
            t0 = time.perf_counter()
            inp = tele.read()
            obs = robot.get_observation()
            now = joints(obs)
            sent = robot.send_action(tele.step(now, inp))

            img = obs["wrist"]
            disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            status = f"REC ep {ep + 1}  {frames} frames" if recording else f"paused  (A = start ep {ep + 1}, L3 = home)"
            cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if recording else (0, 255, 0), 2)
            cv2.imshow("wrist cam", disp); cv2.waitKey(1)

            if recording:
                frame = {
                    **build_dataset_frame(dataset.features, obs, prefix="observation"),
                    **build_dataset_frame(dataset.features, sent, prefix="action"),
                    "task": args.task,
                }
                dataset.add_frame(frame)
                frames += 1

            if inp["B"]:
                if recording:
                    dataset.clear_episode_buffer()
                    print("Discarded unfinished episode.")
                break
            if not recording and inp["A"]:
                tele.set_home(now)  # this pose becomes the start pose; L3 returns here after an episode
                recording, frames = True, 0
                print(f"Recording episode {ep + 1}...")
                time.sleep(0.3)
            elif recording and inp["Y"]:
                dataset.save_episode()
                ep += 1; recording = False
                print(f"Saved episode {ep} ({frames} frames). Move the block, press L3 to go home, then A.")
                time.sleep(0.3)
            elif recording and inp["X"]:
                dataset.clear_episode_buffer()
                recording = False
                print("Episode scrapped. Reset the scene, then press A to redo it.")
                time.sleep(0.3)

            time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - t0)))
    finally:
        print("\nFinalizing dataset (encoding video)...")
        dataset.finalize()
        print(f"Done: {dataset.num_episodes} episodes at {dataset.root}")
        robot.disconnect()
        cv2.destroyAllWindows(); pygame.quit()


if __name__ == "__main__":
    main()
