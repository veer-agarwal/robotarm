"""
Cartesian gamepad control for the SO-101 follower.

Instead of one stick per joint, you move the GRIPPER through space and the script works out
the joint angles (inverse kinematics from the official SO-101 URDF). Shared by drive_xyz.py
and record_gamepad.py.

    Left stick  up/down    : gripper forward / back (away from / toward the base)
    Left stick  left/right : gripper left / right
    Right stick up/down    : gripper up / down
    D-pad up/down          : tilt gripper (pitch) up / down
    LB / RB                : wrist roll
    LT / RT                : gripper close / open
    Left stick click (L3)  : return to the saved start pose
"""

import numpy as np
import pygame

from so101_kin import JOINTS, ik, pos_pitch

# pygame / SDL2 mapping for an Xbox controller on Windows (confirmed with diag.py)
AX_LX, AX_LY, AX_RX, AX_RY, AX_LT, AX_RT = 0, 1, 2, 3, 4, 5
BTN_A, BTN_B, BTN_X, BTN_Y, BTN_LB, BTN_RB, BTN_L3 = 0, 1, 2, 3, 4, 5, 8
DEADZONE = 0.15

# ---- tuning ----
XY_SPEED = 0.004      # metres per tick at full stick (0.004 * 30 Hz = 12 cm/s)
Z_SPEED = 0.003
PITCH_SPEED = 1.5     # degrees per tick (d-pad)
ROLL_SPEED = 3.0      # degrees per tick (bumpers)
GRIP_SPEED = 2.5      # degrees per tick (triggers)
MAX_LEAD = 25.0       # max degrees a joint goal may lead the measured joint (torque headroom)
# Gravity compensation: slowly bias each joint's command until the joint actually reaches
# where the solver put it. KI = fraction of the remaining error added per tick.
GRAV_KI = 0.04
GRAV_MAX = 20.0       # max bias in degrees
GRAV_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")

# ---- workspace box, metres in the base frame (x forward, y left, z up from the base servo) ----
# Loosen/tighten these once you've seen where the desk is in the readout.
REACH_MIN, REACH_MAX = 0.10, 0.32
Z_MIN, Z_MAX = -0.08, 0.30
PITCH_MIN, PITCH_MAX = -95.0, 20.0

# URDF joint limits (degrees, URDF frame) with a small margin
URDF_LIMITS = {
    "shoulder_pan": (-105, 105), "shoulder_lift": (-98, 98), "elbow_flex": (-95, 95),
    "wrist_flex": (-93, 93), "wrist_roll": (-155, 160),
}
TICKS_PER_DEG = 4095 / 360.0


def dz(v: float) -> float:
    return 0.0 if abs(v) < DEADZONE else v


def clamp(v, lo, hi):
    return min(max(v, lo), hi)


class CartesianTeleop:
    """All joint dicts in/out are in LeRobot's calibrated degrees. Internally the solver works in
    the URDF frame: LeRobot's 0 deg is the middle of the range swept at calibration, the URDF's 0
    is raw tick 2048, so each joint differs by a constant offset read from the calibration."""

    def __init__(self, pad: pygame.joystick.Joystick, start_joints: dict, calibration: dict):
        self.pad = pad
        self.off, self.lim = {}, {}
        for j, c in calibration.items():
            mid = (c.range_min + c.range_max) / 2
            self.off[j] = (mid - 2048) / TICKS_PER_DEG          # urdf_deg = lerobot_deg + off
            half = (c.range_max - c.range_min) / 2 / TICKS_PER_DEG
            self.lim[j] = (-half + 1.0, half - 1.0)             # LeRobot frame
        self.lim["wrist_roll"] = (-179, 179)
        print("calibration offsets (deg):", {j: round(v, 1) for j, v in self.off.items()})

        self.roll = start_joints["wrist_roll"]
        self.grip = start_joints["gripper"]
        self.q = {j: float(start_joints[j]) + self.off[j] for j in JOINTS}   # solver state, URDF frame
        p, pitch = pos_pitch([self.q[j] for j in JOINTS])
        self.target = np.array(p, dtype=float)
        self.pitch = float(pitch)
        self.home = dict(start_joints)
        self.homing = False
        self.bias = {j: 0.0 for j in GRAV_JOINTS}

    def _u(self, j, v):   # LeRobot -> URDF
        return v + self.off[j]

    def _l(self, j, v):   # URDF -> LeRobot
        return v - self.off[j]

    def set_home(self, joints: dict) -> None:
        self.home = dict(joints)

    def read(self) -> dict:
        pygame.event.pump()
        pad = self.pad
        hat = pad.get_hat(0) if pad.get_numhats() > 0 else (0, 0)
        return {
            "fwd": -dz(pad.get_axis(AX_LY)),   # stick up = forward
            "left": -dz(pad.get_axis(AX_LX)),  # stick left = +y (left)
            "up": -dz(pad.get_axis(AX_RY)),
            "pitch": float(hat[1]),
            "roll": float(pad.get_button(BTN_RB)) - float(pad.get_button(BTN_LB)),
            "grip": (pad.get_axis(AX_RT) + 1) / 2 - (pad.get_axis(AX_LT) + 1) / 2,
            "home": bool(pad.get_button(BTN_L3)),
            "A": bool(pad.get_button(BTN_A)), "B": bool(pad.get_button(BTN_B)),
            "X": bool(pad.get_button(BTN_X)), "Y": bool(pad.get_button(BTN_Y)),
        }

    def step(self, now: dict, inp: dict) -> dict:
        """now: measured joints (deg). Returns goal dict {joint.pos: deg}."""
        if inp["home"]:
            self.homing = True
        if self.homing:
            # glide toward home; stop homing when close or when the user touches a stick
            if any(abs(inp[k]) > 0 for k in ("fwd", "left", "up")):
                self.homing = False
            else:
                goal = {}
                done = True
                for j in JOINTS + ["gripper"]:
                    g = clamp(self.home[j], now[j] - MAX_LEAD, now[j] + MAX_LEAD)
                    goal[j] = g
                    done &= abs(self.home[j] - now[j]) < 2.0
                if done:
                    self.homing = False
                self._resync(goal)
                return {f"{j}.pos": v for j, v in goal.items()}

        # move the Cartesian target
        self.target[0] += inp["fwd"] * XY_SPEED
        self.target[1] += inp["left"] * XY_SPEED
        self.target[2] += inp["up"] * Z_SPEED
        self.pitch = clamp(self.pitch + inp["pitch"] * PITCH_SPEED, PITCH_MIN, PITCH_MAX)
        self.roll = clamp(self.roll + inp["roll"] * ROLL_SPEED, *self.lim["wrist_roll"])
        self.grip = clamp(self.grip + inp["grip"] * GRIP_SPEED, *self.lim["gripper"])

        # keep the target inside the workspace box
        r = float(np.hypot(self.target[0], self.target[1]))
        if r < 1e-6:
            r = 1e-6
        rc = clamp(r, REACH_MIN, REACH_MAX)
        self.target[0] *= rc / r
        self.target[1] *= rc / r
        self.target[2] = clamp(self.target[2], Z_MIN, Z_MAX)

        # solve joints for the target, warm-started from the previous solution (not the sagging
        # measured pose), so consecutive solutions are smooth
        roll_u = self._u("wrist_roll", self.roll)
        self.q["wrist_roll"] = roll_u
        sol = ik(self.target, self.pitch, self.q, roll_u, iters=20, damping=0.01, step_limit=6)
        for j in JOINTS[:4]:
            lo = max(URDF_LIMITS[j][0], self._u(j, self.lim[j][0]))
            hi = min(URDF_LIMITS[j][1], self._u(j, self.lim[j][1]))
            self.q[j] = clamp(sol[j], lo, hi)

        # if the target is outside what the arm can reach, pull it back to the solver's pose
        p_sol, _ = pos_pitch([self.q[j] for j in JOINTS])
        if np.linalg.norm(p_sol - self.target) > 0.03:
            self.target = np.array(p_sol)

        # gravity compensation: integrate the steady error between where the solver wants each
        # joint and where it actually is
        desired = {j: self._l(j, self.q[j]) for j in JOINTS}
        for j in GRAV_JOINTS:
            self.bias[j] = clamp(self.bias[j] + GRAV_KI * (desired[j] - now[j]), -GRAV_MAX, GRAV_MAX)

        # send (LeRobot frame): solver pose + bias, never more than MAX_LEAD ahead of the measured joint
        goal = {}
        for j in JOINTS:
            g = desired[j] + self.bias.get(j, 0.0)
            goal[j] = clamp(g, now[j] - MAX_LEAD, now[j] + MAX_LEAD)
        goal["gripper"] = self.grip
        return {f"{j}.pos": v for j, v in goal.items()}

    def _resync(self, goal: dict) -> None:
        for j in JOINTS:
            self.q[j] = self._u(j, float(goal[j]))
        p, pitch = pos_pitch([self.q[j] for j in JOINTS])
        self.target = np.array(p); self.pitch = float(pitch)
        self.roll = goal["wrist_roll"]; self.grip = goal["gripper"]

    def status(self) -> str:
        x, y, z = self.target
        b = self.bias.get("shoulder_lift", 0.0)
        return f"x{x*100:5.1f} y{y*100:5.1f} z{z*100:5.1f}cm pitch{self.pitch:5.0f} roll{self.roll:5.0f} grip{self.grip:4.0f} gcomp{b:5.1f}"
