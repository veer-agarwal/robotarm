# SO-101 reinforcement learning in simulation

The real SO-101 (TheRobotStudio's MuJoCo model, Apache-2.0) learning to pick up a 25 mm block from scratch
with SAC, no demonstrations. State-based (joint angles + block pose), joint-delta actions like the gamepad
teleop, 20 Hz control, 5 s episodes, block spawned anywhere in a 10 x 24 cm patch with random yaw.

- `so101_lift_env.py`  Gymnasium env. Finger pads replace the jaw meshes for contact (MuJoCo collides meshes
  as convex hulls, and the fixed jaw's hull fills the finger gap); pads are placed on the measured finger
  surfaces. Contacts stiffened so a 20 g block isn't squeezed through the pads.
- `scripted_check.py`  sanity check: inverse-kinematics script must lift the block (20/20) before RL runs.
- `train_lift.py`      SAC (stable-baselines3), parallel envs, eval every 50k steps on 30 fixed layouts.
- `watch.py`           watch a trained policy in the MuJoCo viewer.

Run: `sim_train.bat` (installs packages, checks the sim, trains 30 h). Watch: `sim_watch.bat`.
Outputs in `runs/sac_lift/`: progress.csv, tensorboard logs (`tensorboard --logdir runs`), best_model.zip,
checkpoints every 250k steps (rerunning sim_train.bat resumes), one eval video per 50k steps.

Next steps after this works: domain randomisation (block size/mass/friction, servo gains, latency),
then a pixel policy from the wrist camera render, then sim-to-real on the arm.
