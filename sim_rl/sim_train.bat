@echo off
rem SO-101 pick-up-the-block reinforcement learning (MuJoCo + SAC). Stops on its own after 30 h.
rem Progress: sim_rl\runs\sac_lift\progress.csv   Videos: sim_rl\runs\sac_lift\videos
cd /d C:\Users\veera\robotarm\sim_rl
set PY=C:\Users\veera\miniforge3\envs\lerobot\python.exe
echo Installing simulator packages into the lerobot env (torch/numpy are pinned so LeRobot keeps working)...
%PY% -m pip install -q -c constraints.txt mujoco==3.17.0 stable-baselines3==2.9.0 tensorboard imageio imageio-ffmpeg
if errorlevel 1 (
  echo.
  echo Package install failed. Take a screenshot of the lines above.
  pause
  exit /b 1
)
%PY% scripted_check.py
if errorlevel 1 (
  echo Sim check failed. Take a screenshot.
  pause
  exit /b 1
)
if not exist runs mkdir runs
echo Training... leave this window open. Log: runs\train_log.txt
if exist runs\sac_lift\checkpoints\*.zip (
  %PY% train_lift.py --hours 30 --resume > runs\train_log.txt 2>&1
) else (
  %PY% train_lift.py --hours 30 > runs\train_log.txt 2>&1
)
echo Finished. Run sim_watch.bat to see the result.
pause
