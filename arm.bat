@echo off
title SO-101 arm control
cd /d C:\Users\veera\robotarm
C:\Users\veera\miniforge3\envs\lerobot\python.exe gamepad_joints.py --port COM8 --id my_follower
echo.
echo Arm control ended. Press any key to close.
pause >nul
