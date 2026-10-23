"""
Joint-space gamepad control for the SO-101 (one stick axis per joint), shared by
gamepad_joints.py and record_gamepad.py.

    Left stick  X : shoulder_pan        Left stick  Y : shoulder_lift
    Right stick X : wrist_flex          Right stick Y : elbow_flex
    LB / RB       : wrist_roll          LT / RT       : gripper close / open
    L3 (click left stick) : glide back to the start pose
"""

import pygame

AX_LX, AX_LY, AX_RX, AX_RY, AX_LT, AX_RT = 0, 1, 2, 3, 4, 5
BTN_A, BTN_B, BTN_X, BTN_Y, BTN_LB, BTN_RB, BTN_L3 = 0, 1, 2, 3, 4, 5, 8
DEADZONE = 0.15

# Degrees the goal moves per tick at full stick (speed).
STEP = {
    "shoulder_pan": 1.5, "shoulder_lift": 1.5, "elbow_flex": 1.5,
    "wrist_flex": 1.5, "wrist_roll": 3.0, "gripper": 2.5,
}
# Flip a sign to reverse a joint's stick direction.
SIGN = {
    "shoulder_pan": 1, "shoulder_lift": -1, "elbow_flex": 1,
    "wrist_flex": 1, "wrist_roll": 1, "gripper": 1,
}
MAX_LEAD = 15.0   # how far the goal may run ahead of the joint while moving (strength)
HOLD_LEAD = 8.0   # how far ahead it may stay once you let go (holds vs gravity, releases at stops)

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def dz(v: float) -> float:
    return 0.0 if abs(v) < DEADZONE else v


def clamp(v, lo, hi):
    return min(max(v, lo), hi)


class JointTeleop:
    def __init__(self, pad: pygame.joystick.Joystick, start_joints: dict):
        self.pad = pad
        self.hold = {j: float(start_joints[j]) for j in JOINTS}
        self.home = dict(start_joints)
        self.homing = False

    def set_home(self, joints: dict) -> None:
        self.home = dict(joints)

    def read(self) -> dict:
        pygame.event.pump()
        pad = self.pad
        lx, ly = dz(pad.get_axis(AX_LX)), dz(pad.get_axis(AX_LY))
        rx, ry = dz(pad.get_axis(AX_RX)), dz(pad.get_axis(AX_RY))
        lt = (pad.get_axis(AX_LT) + 1) / 2
        rt = (pad.get_axis(AX_RT) + 1) / 2
        return {
            "stick": {
                "shoulder_pan": lx, "shoulder_lift": ly, "elbow_flex": ry, "wrist_flex": rx,
                "wrist_roll": float(pad.get_button(BTN_RB)) - float(pad.get_button(BTN_LB)),
                "gripper": rt - lt,
            },
            "home": bool(pad.get_button(BTN_L3)),
            "A": bool(pad.get_button(BTN_A)), "B": bool(pad.get_button(BTN_B)),
            "X": bool(pad.get_button(BTN_X)), "Y": bool(pad.get_button(BTN_Y)),
        }

    def step(self, now: dict, inp: dict) -> dict:
        stick = inp["stick"]
        if inp["home"]:
            self.homing = True
        if self.homing:
            if any(v != 0.0 for v in stick.values()):
                self.homing = False
            else:
                done = True
                for j in JOINTS:
                    self.hold[j] = clamp(self.home[j], now[j] - MAX_LEAD, now[j] + MAX_LEAD)
                    done &= abs(self.home[j] - now[j]) < 2.0
                if done:
                    self.homing = False
                return {f"{j}.pos": v for j, v in self.hold.items()}

        for j in JOINTS:
            p = now[j]
            if stick[j] != 0.0:
                self.hold[j] += SIGN[j] * stick[j] * STEP[j]
                self.hold[j] = clamp(self.hold[j], p - MAX_LEAD, p + MAX_LEAD)
            else:
                self.hold[j] = clamp(self.hold[j], p - HOLD_LEAD, p + HOLD_LEAD)
        return {f"{j}.pos": v for j, v in self.hold.items()}

    def status(self) -> str:
        return "  ".join(f"{j[:5]}:{v:6.1f}" for j, v in self.hold.items())
