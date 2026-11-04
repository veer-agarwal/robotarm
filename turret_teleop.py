"""
Drive the turret with the Xbox controller: pan, tilt and the laser, with the webcam view and a crosshair.

    Left stick        : pan (X) and tilt (Y), full speed       Right stick : same, fine (1/4 speed)
    D-pad             : nudge 0.5 deg                          A           : glide back to centre
    Y                 : ARM / disarm the laser                 RT (hold)   : laser ON while armed
    Back / Start      : flip pan / tilt direction              B           : quit (laser off, torque off)

Usage:  turret_teleop.bat --port COM9 --laser COM11 [--cam 0]
        --laser and --cam are optional: no --cam = no camera window feed (a status screen instead).
Limits: the ones you taught with turret_setup.bat teach (turret_limits.json), else pan +-120, tilt +-40.
Ctrl+C / closing the window / B all stop it; the ESP32 turns the laser off by itself 0.5 s after the PC goes quiet.
"""
import argparse, sys, time
import cv2
import numpy as np
import pygame

from turret_track import Servos, Laser, load_limits, open_camera, PAN_ID, TILT_ID, CENTRE, TPD
from turret_setup import move_watch, ADDR_ACC, ADDR_SPEED

AX_LX, AX_LY, AX_RX, AX_RY, AX_LT, AX_RT = 0, 1, 2, 3, 4, 5
BTN_A, BTN_B, BTN_X, BTN_Y, BTN_LB, BTN_RB, BTN_BACK, BTN_START = 0, 1, 2, 3, 4, 5, 6, 7
DEADZONE = 0.12
PAN_RATE, TILT_RATE = 90.0, 60.0      # deg/s at full left stick
FINE = 0.25                           # right stick speed factor
NUDGE = 0.5                           # deg per D-pad press
HZ = 50


def dz(v):
    if abs(v) < DEADZONE:
        return 0.0
    v = (abs(v) - DEADZONE) / (1 - DEADZONE) * np.sign(v)
    return float(np.sign(v) * abs(v) ** 1.6)          # gentle near the centre, fast at the edge


def hud(frame, pan, tilt, armed, firing, lim, flips):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    col = (0, 0, 255) if firing else (0, 200, 255) if armed else (200, 200, 200)
    cv2.line(frame, (cx - 18, cy), (cx + 18, cy), col, 2); cv2.line(frame, (cx, cy - 18), (cx, cy + 18), col, 2)
    if firing:
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
    (plo, phi), (tlo, thi) = lim[PAN_ID], lim[TILT_ID]
    lines = [f"pan {pan:+6.1f}  [{plo:+.0f} .. {phi:+.0f}]", f"tilt {tilt:+6.1f}  [{tlo:+.0f} .. {thi:+.0f}]",
             "LASER " + ("FIRING" if firing else "ARMED (RT fires)" if armed else "safe (Y to arm)"),
             f"flip pan {'on' if flips[0] < 0 else 'off'}  flip tilt {'on' if flips[1] < 0 else 'off'}   A centre  B quit"]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, t, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col if i == 2 else (255, 255, 255), 1)
    # limit bars
    for i, (v, lo, hi) in enumerate(((pan, plo, phi), (tilt, tlo, thi))):
        y = h - 30 + 14 * i
        cv2.rectangle(frame, (10, y), (210, y + 8), (80, 80, 80), 1)
        x = int(10 + 200 * (v - lo) / max(hi - lo, 1e-6))
        cv2.rectangle(frame, (x - 2, y), (x + 2, y + 8), (0, 255, 0), -1)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="turret servo board (CH343), e.g. COM9")
    ap.add_argument("--laser", help="ESP32 running turret_laser.ino, e.g. COM10 (optional)")
    ap.add_argument("--cam", type=int, default=-1, help="webcam index (e.g. 0); default: no camera")
    ap.add_argument("--width", type=int, default=640); ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--exposure", type=float, default=-5)
    ap.add_argument("--flip-pan", action="store_true"); ap.add_argument("--flip-tilt", action="store_true")
    a = ap.parse_args()

    pygame.init(); pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        sys.exit("No controller found. Plug it in (or pair it) and rerun.")
    pad = pygame.joystick.Joystick(0); pad.init()
    print(f"controller: {pad.get_name()}")

    lim = load_limits()
    servos = Servos(a.port)
    laser = Laser(a.laser) if a.laser else None
    cap = None
    if a.cam >= 0:
        try:
            cap = open_camera(a.cam, a.width, a.height, 30, a.exposure)
        except SystemExit as e:
            print(f"{e}\nNo camera: carrying on without it.")

    print("Gliding to centre...")
    for sid in (PAN_ID, TILT_ID):
        move_watch(servos.bus, sid, CENTRE, speed=300)
        servos.bus.write(sid, ADDR_ACC, 40)
        servos.bus.write(sid, ADDR_SPEED, 2500, 2)
    pan = tilt = 0.0
    flips = [-1 if a.flip_pan else 1, -1 if a.flip_tilt else 1]
    armed = False
    prev = {}
    homing = False
    print("Ready. Y arms the laser, RT fires, B quits.")

    def pressed(b):                      # rising edge
        now = bool(pad.get_button(b)); was = prev.get(b, False); prev[b] = now
        return now and not was

    try:
        last = time.perf_counter()
        while True:
            pygame.event.pump()
            now = time.perf_counter(); dt = min(now - last, 0.1); last = now

            if pressed(BTN_B):
                break
            if pressed(BTN_Y):
                armed = not armed
            if pressed(BTN_BACK):
                flips[0] *= -1
            if pressed(BTN_START):
                flips[1] *= -1
            if pressed(BTN_A):
                homing = True

            lx, ly, rx, ry = (dz(pad.get_axis(i)) for i in (AX_LX, AX_LY, AX_RX, AX_RY))
            vx = lx + FINE * rx
            vy = -(ly + FINE * ry)                       # stick up = tilt up
            hx, hy = pad.get_hat(0) if pad.get_numhats() else (0, 0)
            if homing and (vx or vy or hx or hy):
                homing = False                           # any stick input cancels the glide home
            if homing:
                pan *= 0.9 if abs(pan) > 0.3 else 0; tilt *= 0.9 if abs(tilt) > 0.3 else 0
                homing = bool(pan or tilt)
            else:
                pan += flips[0] * vx * PAN_RATE * dt
                tilt += flips[1] * vy * TILT_RATE * dt
                if hx or hy:
                    key = (hx, hy)
                    if prev.get("hat") != key:
                        pan += flips[0] * NUDGE * hx; tilt += flips[1] * NUDGE * hy
                    prev["hat"] = key
                else:
                    prev["hat"] = None
            pan = float(np.clip(pan, *lim[PAN_ID])); tilt = float(np.clip(tilt, *lim[TILT_ID]))
            servos.goto(pan, tilt)

            rt = (pad.get_axis(AX_RT) + 1) / 2 if pad.get_numaxes() > AX_RT else 0.0
            firing = armed and rt > 0.5
            if laser:
                laser.set(firing)

            frame = None
            if cap is not None:
                ok, frame = cap.read()
                frame = frame if ok else None
            if frame is None:
                frame = np.zeros((360, 640, 3), np.uint8)
            cv2.imshow("turret teleop (B or Q quits)", hud(frame, pan, tilt, armed, firing, lim, flips))
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
            if cv2.getWindowProperty("turret teleop (B or Q quits)", cv2.WND_PROP_VISIBLE) < 1:
                break
            time.sleep(max(0.0, 1 / HZ - (time.perf_counter() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping: laser off, gliding to centre, torque off.")
        if laser:
            laser.close()
        try:
            for sid in (PAN_ID, TILT_ID):
                move_watch(servos.bus, sid, CENTRE, speed=300)
        except Exception:
            pass
        servos.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows(); pygame.quit()


if __name__ == "__main__":
    main()
