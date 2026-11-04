"""
Tello: live camera on the PC, with optional Xbox / keyboard flying. Step 1 toward CV + pathfinding on the PC.

Setup (once):  C:\\Users\\veera\\miniforge3\\envs\\lerobot\\python.exe -m pip install djitellopy
Then: turn the Tello on, join its Wi-Fi (TELLO-xxxxxx) on the PC, and run
    tello_fly.bat                 camera only, the motors never start (default)
    tello_fly.bat --fly           camera + flying with the controller / keyboard
    tello_fly.bat --record        also save every 5th frame to datasets\\tello_cam\\ (for training CV later)
When Windows asks about the firewall the first time: allow Python on PRIVATE networks, or there's no video.

Flying (--fly), Xbox controller (mode 2, like a normal drone remote):
    Start       : take off                     B / Esc / Q : land (and quit with Esc / Q)
    Left stick  : up/down + turn               Right stick : forward/back + left/right
    LB (hold)   : slow mode (30 %)             Back        : EMERGENCY motor stop (falls! only if it's about to hit something)
Keyboard (no controller): T take off, L land, W/S forward/back, A/D left/right, R/F up/down, Z/C turn, Esc quit.
It lands by itself under 15 % battery, and lands on any crash/quit.
"""
import argparse, os, time
import cv2
import numpy as np

AX_LX, AX_LY, AX_RX, AX_RY = 0, 1, 2, 3
BTN_B, BTN_LB, BTN_BACK, BTN_START = 1, 4, 6, 7
DEADZONE, SPEED, SLOW = 0.12, 70, 0.3          # rc values are -100..100; 70 is plenty indoors
LOW_BATT = 15


def dz(v):
    return 0.0 if abs(v) < DEADZONE else (abs(v) - DEADZONE) / (1 - DEADZONE) * np.sign(v)


def hud(img, t, fly, flying, fps, rec):
    try:
        b, h, tmp = t.get_battery(), t.get_height(), t.get_temperature()
    except Exception:
        b = h = tmp = -1
    col = (0, 0, 255) if 0 <= b < 25 else (255, 255, 255)
    lines = [f"battery {b}%   height {h} cm   temp {tmp:.0f} C   {fps:4.1f} fps",
             ("FLYING" if flying else "landed") + ("" if fly else "   (camera only: run with --fly to fly)") + ("   REC" if rec else "")]
    for i, s in enumerate(lines):
        cv2.putText(img, s, (10, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(img, s, (10, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 1)
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fly", action="store_true", help="allow take-off (otherwise camera only)")
    ap.add_argument("--record", action="store_true", help="save every 5th frame for a CV dataset")
    a = ap.parse_args()

    from djitellopy import Tello
    t = Tello()
    try:
        t.connect()
    except Exception as e:
        raise SystemExit(f"Can't reach the Tello ({e}).\nIs the PC on the TELLO-xxxxxx Wi-Fi? Is the Tello on (yellow blinking)? "
                         "Close the Tello phone app if it's open.")
    print(f"Connected. Battery {t.get_battery()}%")
    t.streamoff(); t.streamon()
    reader = t.get_frame_read()

    pad = None
    if a.fly:
        import pygame
        pygame.init(); pygame.joystick.init()
        if pygame.joystick.get_count():
            pad = pygame.joystick.Joystick(0); pad.init(); print(f"controller: {pad.get_name()}")
        else:
            print("No controller: keyboard only (click the video window first).")

    out = None
    if a.record:
        out = os.path.join("datasets", "tello_cam", time.strftime("%Y%m%d_%H%M%S")); os.makedirs(out, exist_ok=True)
        print(f"recording to {out}")

    flying, n, last, fps, prev = False, 0, time.perf_counter(), 0.0, {}
    win = "tello (Esc quits)"

    def pressed(b):
        now = bool(pad.get_button(b)); was = prev.get(b, False); prev[b] = now
        return now and not was

    try:
        while True:
            frame = reader.frame
            if frame is None:
                time.sleep(0.01); continue
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)          # djitellopy hands frames over as RGB
            now = time.perf_counter(); fps = 0.9 * fps + 0.1 / max(now - last, 1e-3); last = now
            n += 1
            if out and n % 5 == 0:
                cv2.imwrite(os.path.join(out, f"{n:06d}.jpg"), img)
            batt = hud(img, t, a.fly, flying, fps, bool(out))
            cv2.imshow(win, img)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break

            if not a.fly:
                continue
            lr = fb = ud = yaw = 0
            if pad is not None:
                import pygame
                pygame.event.pump()
                s = SPEED * (SLOW if pad.get_button(BTN_LB) else 1.0)
                yaw, ud = dz(pad.get_axis(AX_LX)) * s, -dz(pad.get_axis(AX_LY)) * s
                lr, fb = dz(pad.get_axis(AX_RX)) * s, -dz(pad.get_axis(AX_RY)) * s
                if pressed(BTN_BACK) and flying:
                    t.emergency(); flying = False; print("EMERGENCY STOP"); continue
                if pressed(BTN_START) and not flying:
                    k = ord("t")
                if pressed(BTN_B) and flying:
                    k = ord("l")
            keymap = {ord("w"): (0, 1, 0, 0), ord("s"): (0, -1, 0, 0), ord("a"): (-1, 0, 0, 0), ord("d"): (1, 0, 0, 0),
                      ord("r"): (0, 0, 1, 0), ord("f"): (0, 0, -1, 0), ord("z"): (0, 0, 0, -1), ord("c"): (0, 0, 0, 1)}
            if k in keymap:
                lr, fb, ud, yaw = (v * SPEED * 0.6 for v in keymap[k])
            if k == ord("t") and not flying:
                if 0 <= batt < LOW_BATT + 10:
                    print("Battery too low to take off."); continue
                t.takeoff(); flying = True
            elif k == ord("l") and flying:
                t.land(); flying = False
            elif flying and 0 <= batt < LOW_BATT:
                print("Low battery: landing."); t.land(); flying = False
            elif flying:
                t.send_rc_control(int(lr), int(fb), int(ud), int(yaw))   # also keeps the Tello from timing out
    finally:
        try:
            if flying:
                t.send_rc_control(0, 0, 0, 0); t.land()
        except Exception:
            pass
        try:
            t.streamoff()
        except Exception:
            pass
        t.end(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
