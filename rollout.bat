@echo off
rem Run a trained pick-the-block policy on the arm (60 s per run, Ctrl+C stops early).
rem   rollout.bat                -> act_aug_c100 (default)
rem   rollout.bat act_aug_c50 | act | diffusion | diffusion_aug | smolvla | smolvla_aug
cd /d C:\Users\veera\robotarm
set M=%1
if "%M%"=="" set M=act_aug_c100
set P=outputs\train\%M%_pick_block\checkpoints\last\pretrained_model
if not exist %P% (echo No model at %P% & pause & exit /b 1)
set R=C:\Users\veera\miniforge3\envs\lerobot\Scripts\lerobot-rollout.exe
echo Running %M%. Hand on the 12 V plug.
if /i "%M:~0,7%"=="smolvla" (
  %R% --strategy.type=base --policy.path=%P% --robot.type=so101_follower --robot.port=COM8 --robot.id=my_follower --robot.max_relative_target=12 --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --rename_map="{observation.images.wrist: observation.images.camera1}" --task="Pick up the block" --duration=60
) else (
  %R% --strategy.type=base --policy.path=%P% --robot.type=so101_follower --robot.port=COM8 --robot.id=my_follower --robot.max_relative_target=12 --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --task="Pick up the block" --duration=60
)
pause
