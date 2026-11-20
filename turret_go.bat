@echo off
rem Drone tracking with the trained detector + laser, with the measured laser corrections.
rem Extra flags pass through, e.g.  turret_go.bat --exposure -7
cd /d C:\Users\veera\robotarm
C:\Users\veera\miniforge3\envs\lerobot\python.exe turret_track.py --cam 0 --port COM9 --mode predict --width 1280 --height 720 --fps 120 --fisheye --focal 641 --exposure -6 --lead 0.03 --detect --nano COM11 --muzzle-below 0.015 --muzzle-right -0.043 --laser-trim-down 2.38 --laser-trim-right 2.59 %*
