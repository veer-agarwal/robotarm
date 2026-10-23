# robotarm - SO-101 follower arm with Xbox joint control

Code and print files for a LeRobot SO-101 follower arm (6x Feetech STS3215, 12 V), driven in joint space from an Xbox controller, with a wrist camera, dataset recording and ACT training on Windows.

## Layout
- `gamepad_joints.py` - drive the arm (`arm.bat` launches it). `joint_control.py` holds the stick mapping, `arm_common.py` the safe connect.
- `record_gamepad.py` - record LeRobot datasets with the controller (A start, Y save, X scrap, B quit).
- `cams.py` - preview camera indices.
- `so101_kin.py`, `xyz_control.py`, `drive_xyz.py` - Cartesian control experiment (unused).
- `diag.py`, `motor_health.py`, `midtest.py`, `holdtest.py`, `regdump.py`, `rolltest.py` - servo diagnostics.
- `calibration/` - follower calibration json.
- `stl/` - every part to print, one folder per part set (see `stl/README.md`).
- `gripper/` - design scripts, previews and notes for the hot-swap gripper socket and Tello launch pad.
- `outputs/` - datasets/checkpoints, ignored by git.

## Run
```
C:\Users\veera\miniforge3\envs\lerobot\python.exe gamepad_joints.py --port COM8 --id my_follower
```
