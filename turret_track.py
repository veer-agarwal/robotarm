"""Turret tracker: camera -> target -> pan/tilt servos (IDs 7/8) -> laser (Nano).

    python turret_track.py --cam 0                       camera only (no servos), test the view + tracker
    python turret_track.py --cam 0 --port COM9           drive the servos too
    python turret_track.py --cam 0 --port COM9 --nano COM10   plus the laser

    python turret_track.py --cam 1 --port COM9 --mode predict --dfov 130   fast drone tracking (Arducam)

Modes (F cycles):
  fixed    camera sits still on the desk; the servos point where the target is in the picture
           (for testing before the camera is mounted on the turret)
  follow   camera rides on the turret; the servos move to keep the target in the centre
  predict  camera rides on the turret; works out where the target is in the room, how fast it is moving,
           and aims where it will be (turret_predict.py). Use this one for the drone. Reads the servo angles each
           frame, so it needs --port. Set --hfov (or --dfov) to your lens.

Keys:  C  click a new target (draw a box, Enter)    X  drop the target    F  next mode
       R  start/stop recording raw video to recordings\\  (for drone clips -> turret_label.py)
       L  arm/disarm the laser. It fires only in the FIRE state: locked 150 ms, aim error under 0.5 deg,
          prediction steady (turret_hud.py has the rule and the HUD)
       P / T  flip pan / tilt direction      E / Shift+E  exposure darker / brighter   A  auto exposure   Q  quit
"""
import argparse, math, os, sys, time
from collections import deque
import cv2
import numpy as np

TPD = 4096 / 360.0                   # servo ticks per degree
CENTRE = 2048
PAN_ID, TILT_ID = 7, 8
PAN_LIM, TILT_LIM = 120.0, 40.0      # degrees either side of centre (cable loop / payload clearance)


LIMITS = {}


def _limits_file():
    import json
    try:
        return json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "turret_limits.json")))
    except Exception:
        return {}


def load_home():
    """Home (the 0 deg aim) per servo in ticks, set with turret_setup.py sethome; default the servo centre 2048."""
    h = _limits_file().get("home", {})
    return {sid: int(h.get(str(sid), CENTRE)) for sid in (PAN_ID, TILT_ID)}


def load_limits():
    """Limits taught with turret_setup.py teach (servo ticks), as degrees from HOME; else the defaults above."""
    d, home = _limits_file(), load_home()
    rng = {PAN_ID: (-PAN_LIM, PAN_LIM), TILT_ID: (-TILT_LIM, TILT_LIM)}
    for sid in rng:
        if str(sid) in d:
            lo, hi = d[str(sid)]
            rng[sid] = ((lo - home[sid]) / TPD + 0.5, (hi - home[sid]) / TPD - 0.5)
        else:
            c = (CENTRE - home[sid]) / TPD
            rng[sid] = (rng[sid][0] + c, rng[sid][1] + c)
    return rng


def make_tracker():
    for mod in (cv2, getattr(cv2, "legacy", None)):
        if mod is None:
            continue
        for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
            f = getattr(mod, name, None)
            if f:
                return f(), name.replace("Tracker", "").replace("_create", "")
    sys.exit("No OpenCV tracker available. Run: pip install opencv-contrib-python")


def open_camera(idx, w, h, fps, exposure):
    from camopen import open_capture
    cap, backend = open_capture(idx, w, h, fps)
    if cap is None:
        sys.exit(f"Camera {idx} didn't open. Try --cam 0 (or 1, 2), and close other apps using it (e.g. the Camera app).")
    from camopen import set_exposure
    if not set_exposure(cap, exposure):
        print(f"(camera ignored the fixed exposure {exposure}; it stays on auto)")
    print(f"camera {idx}: {int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5):.0f} fps requested, exposure {exposure}")
    return cap


class Servos:
    def __init__(self, port, acc=60):
        from turret_setup import Bus, ADDR_TORQUE, ADDR_ACC, ADDR_SPEED, ADDR_GOAL, ADDR_POS, WRITE
        self.bus, self.GOAL, self.WRITE, self.POS = Bus(port), ADDR_GOAL, WRITE, ADDR_POS
        for sid in (PAN_ID, TILT_ID):
            if not self.bus.ping(sid):
                sys.exit(f"No servo {sid} on {port}. Run turret_setup.bat scan --port {port}.")
            self.bus.write(sid, ADDR_ACC, acc)            # 60 = gentle, 254 = snappy (what predict mode wants)
            self.bus.write(sid, ADDR_SPEED, 2500, 2)
            self.bus.write(sid, ADDR_TORQUE, 1)
        self.ADDR_TORQUE = ADDR_TORQUE
        self.home = load_home()

    def goto(self, pan_deg, tilt_deg):
        for sid, deg in ((PAN_ID, pan_deg), (TILT_ID, tilt_deg)):
            t = int(round(self.home[sid] + deg * TPD))
            self.bus._tx(sid, self.WRITE, [self.GOAL] + list(t.to_bytes(2, "little")))
            self.bus._rx(sid, 0)

    def read_deg(self):
        """Present (pan, tilt) in degrees from home, or None if a read failed."""
        out = []
        for sid in (PAN_ID, TILT_ID):
            v = self.bus.read(sid, self.POS, 2)
            if v is None: return None
            if v & 0x8000: v = -(v & 0x7FFF)              # sign bit
            out.append((v - self.home[sid]) / TPD)
        return tuple(out)

    def release(self):
        for sid in (PAN_ID, TILT_ID):
            try:
                self.bus._tx(sid, self.WRITE, [self.ADDR_TORQUE, 0]); self.bus._rx(sid, 0)
            except Exception:
                pass
        self.bus.close()


SPINUP, REV_HOLD = 0.3, 0.6          # blaster: flywheel spin-up before the pusher runs; keep spinning after a burst
TARGET_SIZE = 0.20                   # Tello with prop guards, metres (used to estimate range from the box size)


class Laser:
    """Nano running turret_laser.ino: '1' = on, '0' = off, turns itself off if nothing arrives for 0.5 s."""
    def __init__(self, port):
        import serial
        try:
            self.s = serial.Serial(port, 115200, timeout=0)
        except serial.SerialException as e:
            sys.exit(f"Can't open the laser port {port} ({e}).\n"
                     "If it says 'Access is denied', another program has it open: close the Arduino IDE Serial Monitor and retry.")
        time.sleep(2.0)                  # Nano resets when the port opens
        self.last = 0.0

        self.blaster = False; self.rev = False; self.fire = False; self.rev_since = 0.0; self.fire_until = 0.0

    def set(self, on, blaster_fire=False):
        """Called every frame. on = laser. blaster_fire = FIRE state while armed (only used with --blaster):
        flywheels rev, the pusher runs once they've spun up (the ESP32 enforces that too), and the flywheels keep
        spinning for REV_HOLD s after the last shot so the next burst is instant."""
        now = time.time()
        if self.blaster:
            if blaster_fire:
                if not self.rev: self.rev, self.rev_since = True, now
                self.fire_until = now + REV_HOLD
                self.fire = now - self.rev_since >= SPINUP
            else:
                self.fire = False
                if self.rev and now > self.fire_until: self.rev = False
        if now - self.last > 0.1:
            msg = b"1" if on else b"0"
            if self.blaster: msg += (b"R" if self.rev else b"r") + (b"F" if self.fire else b"f")
            self.s.write(msg); self.last = now

    def close(self):
        try: self.s.write(b"0rf"); self.s.close()
        except Exception: pass


def angle_at(hist, t):
    """Servo (pan, tilt) at time t, interpolated from the timestamped reads in hist."""
    if t <= hist[0][0]: return hist[0][1], hist[0][2]
    for (t0, p0, q0), (t1, p1, q1) in zip(hist, list(hist)[1:]):
        if t0 <= t <= t1:
            w = (t - t0) / max(1e-6, t1 - t0)
            return p0 + w * (p1 - p0), q0 + w * (q1 - q0)
    return hist[-1][1], hist[-1][2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--exposure", type=float, default=-6, help="DirectShow log2 seconds: -6 = 1/64 s, -8 = 1/256 s")
    ap.add_argument("--hfov", type=float, default=70.0, help="camera horizontal field of view, degrees")
    ap.add_argument("--focal", type=float, help="focal length in pixels (turret_cam_check --fov measures it; Arducam at 1280x720: 641)")
    ap.add_argument("--fisheye", action="store_true", help="wide-angle lens (Arducam 130 deg): angle grows evenly toward the edges")
    ap.add_argument("--dfov", type=float, help="diagonal field of view instead (Arducam 130 deg lens: --dfov 130)")
    ap.add_argument("--port", help="turret servo board, e.g. COM9 (omit for camera-only)")
    ap.add_argument("--nano", help="Nano for the laser, e.g. COM10")
    ap.add_argument("--mode", choices=["fixed", "follow", "predict"], default="fixed")
    ap.add_argument("--kp", type=float, default=0.35, help="follow mode: fraction of the error corrected per frame")
    ap.add_argument("--tracker", choices=["csrt", "auto", "blob", "template"], default="csrt",
                    help="predict mode: csrt (default; best in clutter, re-finds the target after a slip); auto = blob for a"
                         " target darker than the background, else template (fastest, fine against plain walls)")
    ap.add_argument("--lead", type=float, default=0.03, help="predict mode: seconds to aim ahead (sim best: 0.03)")
    ap.add_argument("--q", type=float, default=2e6, help="predict mode: how fast the target can change direction")
    ap.add_argument("--latency", type=float, default=0.025, help="predict mode: camera delay, seconds (capture -> frame here)")
    ap.add_argument("--acc", type=int, help="servo ACC register 0-254 (default 60; 254 in predict mode)")
    ap.add_argument("--muzzle-below", type=float, default=0.033,
                    help="predict mode: how far the laser/muzzle sits BELOW the camera lens, metres (laser plate: 0.033)")
    ap.add_argument("--muzzle-right", type=float, default=0.0, help="...and to the RIGHT of it, metres")
    ap.add_argument("--dart-speed", type=float, default=0.0,
                    help="m/s; 0 = laser (no drop). With darts (~30) it aims up for drop and leads for flight time")
    ap.add_argument("--mask", help="ignore part of the picture, fractions x0,y0,x1,y1 (e.g. the laser block: 0.4,0.85,0.6,1)")
    ap.add_argument("--blaster", action="store_true", help="the --nano ESP32 also drives the rev/fire relays")
    ap.add_argument("--log", action="store_true", help="predict mode: write recordings/track_log_<time>.csv for tuning")
    a = ap.parse_args()
    if a.mode == "predict" and not a.port:
        sys.exit("predict mode reads the servo angles, so it needs --port (e.g. --port COM9).")
    global LIMITS
    LIMITS = load_limits()
    print(f"limits: pan {LIMITS[PAN_ID][0]:+.0f}..{LIMITS[PAN_ID][1]:+.0f} deg, tilt {LIMITS[TILT_ID][0]:+.0f}..{LIMITS[TILT_ID][1]:+.0f} deg")

    cap = open_camera(a.cam, a.width, a.height, a.fps, a.exposure)
    servos = Servos(a.port, a.acc if a.acc is not None else (254 if a.mode == "predict" else 60)) if a.port else None
    laser = Laser(a.nano) if a.nano else None
    if a.blaster:
        if not laser: sys.exit("--blaster needs --nano (the ESP32's port)")
        laser.blaster = True
    mask = [float(v) for v in a.mask.split(",")] if a.mask else None
    rng_est = None                                   # filtered range to the target, metres
    log = None
    if a.log:
        os.makedirs("recordings", exist_ok=True)
        log = open(os.path.join("recordings", time.strftime("track_log_%Y%m%d_%H%M%S.csv")), "w", buffering=1)
        log.write("t,t_cap,pan_meas,tilt_meas,found,box_x,box_y,box_w,box_h,kf_pan,kf_pan_v,kf_tilt,kf_tilt_v,goal_pan,goal_tilt,"
                  "servo_pan_cmd,servo_tilt_cmd,servo_pan_now,servo_tilt_now\n")
    os.makedirs("recordings", exist_ok=True)

    tracker, tname, box = None, "", None
    aim = None                                       # predict mode's Predictive
    angle_hist = deque(maxlen=64)                    # (time, pan, tilt) as read from the servos
    lens = None
    # Directions, checked from a predict-mode log (Nov 10): pan servo + turns the camera RIGHT, tilt servo + turns it UP.
    # (The old -1 for pan made predict mode chase its own tail: turning toward the target moved it further away.)
    pan = tilt = 0.0; pan_sign, tilt_sign = 1.0, -1.0
    mode, armed, writer, exposure = a.mode, False, None, a.exposure
    t_prev, fps_est = time.time(), 0.0
    import turret_hud
    fc = turret_hud.FireControl(); seen = 0             # fire rule; frames in a row with the target actually seen
    win = "turret"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if servos: servos.goto(0, 0)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera stopped delivering frames"); break
            t_frame = time.perf_counter()
            if mask is not None:                     # blank out a part of the picture the tracker should ignore
                Hm, Wm = frame.shape[:2]
                x0, y0, x1, y1 = int(mask[0] * Wm), int(mask[1] * Hm), int(mask[2] * Wm), int(mask[3] * Hm)
                frame[y0:y1, x0:x1] = np.median(frame, axis=(0, 1)).astype(frame.dtype)
            raw = frame.copy()                       # clean copy: the target is picked from this, not the overlay
            H, W = frame.shape[:2]
            vfov = a.hfov * H / W
            if lens is None or (lens.w, lens.h) != (W, H):
                from turret_predict import Lens
                lens = Lens(W, H, hfov=a.hfov, diag_fov=a.dfov, focal=a.focal, fisheye=a.fisheye)
            if mode == "predict" and servos:
                r = servos.read_deg()
                if r is not None:                                 # store in the predictor's convention (+ = left / up)
                    angle_hist.append((time.perf_counter(), -pan_sign * r[0], -tilt_sign * r[1]))
            if writer is not None: writer.write(frame)
            now = time.time(); fps_est = 0.9 * fps_est + 0.1 / max(1e-3, now - t_prev); t_prev = now

            locked = False
            hud = dict(box=None, lead=None, err=None, rate=None, range=None); tracking = steady = found_now = False
            if mode == "predict" and aim is not None and angle_hist:
                t_cap = t_frame - a.latency
                pm, tm = angle_at(angle_hist, t_cap)               # where the turret pointed when the frame was taken
                flight = (rng_est / a.dart_speed) if (rng_est and a.dart_speed > 0) else 0.0
                aim.lead = a.lead + flight                          # darts: also lead by their flight time
                goal, locked = aim.update(frame, t_cap, pm, tm, time.perf_counter())
                if goal is not None and max(abs(aim.kp.x[1]), abs(aim.kt.x[1])) > 300:
                    goal = None                                      # runaway (target "moving" > 300 deg/s): let go
                    print("predict: target estimate ran away, dropped it - press C to pick again")
                if aim is not None and aim.box is not None:          # range from how big the drone looks
                    half_px = max(aim.box[2], aim.box[3]) / 2
                    if half_px > 1:
                        r_now = (TARGET_SIZE / 2) * lens.f / half_px
                        rng_est = r_now if rng_est is None else 0.8 * rng_est + 0.2 * r_now
                corr_p = corr_t = 0.0                               # muzzle isn't on the camera axis: aim it, not the lens
                if rng_est:
                    corr_t = math.degrees(math.atan2(a.muzzle_below, rng_est))
                    corr_p = math.degrees(math.atan2(a.muzzle_right, rng_est))
                    if a.dart_speed > 0:
                        corr_t += math.degrees(math.atan2(0.5 * 9.81 * flight ** 2, rng_est))
                if goal is not None:
                    goal = (goal[0] + corr_p, goal[1] + corr_t)
                if goal is None:
                    aim = None
                else:
                    pan = float(np.clip(-pan_sign * goal[0], *LIMITS[PAN_ID]))
                    tilt = float(np.clip(-tilt_sign * goal[1], *LIMITS[TILT_ID]))
                    servos.goto(pan, tilt)
                    if log:
                        b_ = aim.box if aim.box is not None else (float("nan"),) * 4
                        _, pn_, tn_ = angle_hist[-1]
                        log.write(",".join(f"{v:.4f}" for v in (
                            time.perf_counter(), t_cap, pm, tm, float(aim.misses == 0), *b_, aim.kp.x[0], aim.kp.x[1],
                            aim.kt.x[0], aim.kt.x[1], goal[0], goal[1], pan, tilt, pn_, tn_)) + "\n")
                    from turret_predict import world_to_pixel
                    t_n = time.perf_counter()
                    _, p_now, t_now_ = angle_hist[-1]
                    bp, bt = aim.predicted(t_n)                     # where the target is now vs where the turret points
                    bp, bt = bp + corr_p, bt + corr_t               # where the MUZZLE has to point
                    hud["err"] = math.hypot((bp - p_now) * math.cos(math.radians(bt)), bt - t_now_)
                    hud["range"] = rng_est
                    hud["rate"] = math.hypot(aim.kp.x[1], aim.kt.x[1])
                    hud["box"] = aim.box
                    hud["lead"] = world_to_pixel(lens, pm, tm, *aim.predicted(t_n + a.lead))
                    tracking = True
                    tracking = True; found_now = aim.misses == 0
            elif mode != "predict" and tracker is not None:
                found, b = tracker.update(frame)
                if found:
                    box = tuple(int(v) for v in b)
                    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
                    ex = (cx - W / 2) / W * a.hfov            # degrees right of centre
                    ey = (cy - H / 2) / H * vfov              # degrees below centre
                    if mode == "fixed":                       # point at it (camera static)
                        pan = 0.6 * pan + 0.4 * (pan_sign * ex)
                        tilt = 0.6 * tilt + 0.4 * (tilt_sign * ey)
                    else:                                     # keep it centred (camera on the turret)
                        pan += np.clip(pan_sign * a.kp * ex, -4, 4)
                        tilt += np.clip(tilt_sign * a.kp * ey, -4, 4)
                    pan = float(np.clip(pan, *LIMITS[PAN_ID])); tilt = float(np.clip(tilt, *LIMITS[TILT_ID]))
                    if servos: servos.goto(pan, tilt)
                    hud["box"] = box; tracking = True; found_now = True
                    if mode == "follow":                      # aim error is only known when the camera rides along
                        hud["err"] = math.hypot(ex, ey)
            seen = seen + 1 if tracking and found_now else 0
            steady = seen >= 6                             # seen in 6 frames straight (~50 ms at 120 fps)

            half = None                                     # target's half-width in degrees, from its box
            if hud["box"] is not None and lens is not None:
                size = aim.size_px() if (mode == "predict" and aim is not None) else max(hud["box"][2], hud["box"][3])
                half = math.degrees(math.atan(size / 2 / lens.f))
            state = fc.update(tracking, hud["err"], steady, armed, half)
            if laser: laser.set(state == "FIRE", blaster_fire=(state == "FIRE" and armed))

            # HUD (drawn on a scaled copy; recordings stay clean)
            hud.update(state=state, reason=fc.reason, hold_ms=fc.hold_ms, armed=armed, pan=pan, tilt=tilt, fps=fps_est,
                       tracker=tname.replace("predictive ", "").split(",")[0].strip("()").upper() if tname else "",
                       mode=mode, recording=writer is not None)
            cv2.imshow(win, turret_hud.draw(frame, hud, out_w=max(1280, W)))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27): break
            elif k == ord("c"):
                r = cv2.selectROI(win, raw, showCrosshair=True)
                if r[2] > 4 and r[3] > 4:
                    if mode == "predict":
                        from turret_predict import Predictive
                        if angle_hist:
                            _, pm, tm = angle_hist[-1]
                            aim = Predictive(lens, a.tracker); aim.start(raw, r, pm, tm); tname = aim.name; rng_est = None
                    else:
                        tracker, tname = make_tracker(); tracker.init(raw, r); box = r
            elif k == ord("x"): tracker = None; box = None; aim = None
            elif k == ord("f"):
                order = ["fixed", "follow"] + (["predict"] if servos else [])
                mode = order[(order.index(mode) + 1) % len(order)]; tracker = None; aim = None
                if mode == "predict":
                    print("predict mode: press C to pick the target. ACC is", "set by --acc" if a.acc is not None else "60 here;"
                          " start with --mode predict for ACC 254")
            elif k == ord("p"): pan_sign = -pan_sign
            elif k == ord("t"): tilt_sign = -tilt_sign
            elif k == ord("l"): armed = not armed
            elif k in (ord("e"), ord("E")):
                exposure += -1 if k == ord("e") else 1
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25); cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            elif k == ord("a"):                       # back to the camera's automatic exposure
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75); exposure = float("nan")
            elif k == ord("r"):
                if writer is None:
                    fn = os.path.join("recordings", time.strftime("clip_%Y%m%d_%H%M%S.mp4"))
                    writer = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*"mp4v"), max(15, round(fps_est)), (W, H))
                    print("recording", fn)
                else:
                    writer.release(); writer = None; print("recording saved")
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None: writer.release()
        if laser: laser.close()
        if log: log.close(); print("log written to recordings/")
        if servos: servos.goto(0, 0); time.sleep(0.5); servos.release()
        cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
