@echo off
rem SO-101 pick-and-place reinforcement learning (MuJoCo + SAC). Stops on its own after 12 h.
rem Progress: sim_rl\runs\sac_place\progress.csv   Videos: sim_rl\runs\sac_place\videos
cd /d C:\Users\veera\robotarm\sim_rl
set PY=C:\Users\veera\miniforge3\envs\lerobot\python.exe
%PY% scripted_check_place.py
if errorlevel 1 (
  echo Sim check failed. Take a screenshot.
  pause
  exit /b 1
)
if not exist runs mkdir runs
echo Training pick-and-place... leave this window open. Log: runs\train_place_log.txt
if exist runs\sac_place\checkpoints\*.zip (
  %PY% train_lift.py --task place --hours 12 --resume > runs\train_place_log.txt 2>&1
) else (
  %PY% train_lift.py --task place --hours 12 > runs\train_place_log.txt 2>&1
)
echo Finished. Run sim_watch.bat to see the result.
pause
