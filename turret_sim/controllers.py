"""Aiming controllers. Each gets a frame, the time it was captured, and the servo angles read at that moment,
and returns new goal angles (or None to leave the goal alone). Same code shape the real turret_track.py loop uses."""
import math
import numpy as np
import cv2


import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from turret_predict import Predictive, KF1, make_cv_tracker, bgr, pixel_bearing      # the exact code the turret runs


class Baseline:
    """What turret_track.py does today in follow mode: CSRT box, linear pixel->degree, pan += kp*error each frame."""
    name = "current (CSRT + P)"

    def __init__(self, cam, kp=0.35, tracker="CSRT"):
        self.cam, self.kp, self.kind = cam, kp, tracker
        self.trk = None; self.pan = self.tilt = 0.0; self.box = None

    def start(self, frame, box, pan, tilt):
        self.trk = make_cv_tracker(self.kind); self.trk.init(bgr(frame), tuple(int(v) for v in box))
        self.pan, self.tilt, self.box = pan, tilt, box

    def update(self, frame, t_cap, pan_meas, tilt_meas, t_now):
        ok, b = self.trk.update(bgr(frame))
        if not ok:
            self.box = None; return None, False
        self.box = b
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        ex = (cx - self.cam.w / 2) / self.cam.w * self.cam.hfov
        ey = (cy - self.cam.h / 2) / self.cam.h * self.cam.vfov
        self.pan += float(np.clip(-self.kp * ex, -4, 4))       # image right = turret right = pan negative
        self.tilt += float(np.clip(-self.kp * ey, -4, 4))
        locked = abs(ex) < 2.0 and abs(ey) < 2.0
        return (self.pan, self.tilt), locked
