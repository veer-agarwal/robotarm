@echo off
rem Drone tracking + Venom Pro blaster on the blaster cradle (upside down, camera 56.5 mm above the barrel).
rem Aims the BARREL: --muzzle-below 0.0565, dart speed ~28 m/s for drop + flight-time lead.
rem --laser-trim-down / --laser-trim-right become the barrel trims once darts are calibrated (start at 0).
rem L = arm. Starts SAFE. Kill switch on the blaster OFF when not testing.
cd /d C:\Users\veera\robotarm
C:\Users\veera\miniforge3\envs\lerobot\python.exe turret_track.py --cam 0 --port COM9 --mode predict --width 1280 --height 720 --fps 120 --fisheye --focal 641 --exposure -6 --lead 0.03 --detect --nano COM11 --blaster --muzzle-below 0.0565 --muzzle-right 0 --dart-speed 28 --laser-trim-down 0 --laser-trim-right 0 %*
