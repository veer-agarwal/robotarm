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
       L  arm/disarm the laser (fires only when locked on, in follow or predict mode)
       P / T  flip pan / tilt direction      E / Shift+E  exposure darker / brighter   A  auto exposure   Q  quit
"""
import argparse, os, sys, time
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
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
    if not cap.isOpened():
        sys.exit(f"Camera {idx} didn't open. Try --cam 1 (or 2), and close other apps using it.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h); cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)     # manual (DirectShow convention)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
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

    def set(self, on):
        if time.time() - self.last > 0.1:
            self.s.write(b"1" if on else b"0"); self.last = time.time()

    def close(self):
        try: self.s.write(b"0"); self.s.close()
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
    ap.add_argument("--dfov", type=float, help="diagonal field of view instead (Arducam 130 deg lens: --dfov 130)")
    ap.add_argument("--port", help="turret servo board, e.g. COM9 (omit for camera-only)")
    ap.add_argument("--nano", help="Nano for the laser, e.g. COM10")
    ap.add_argument("--mode", choices=["fixed", "follow", "predict"], default="fixed")
    ap.add_argument("--kp", type=float, default=0.35, help="follow mode: fraction of the error corrected per frame")
    ap.add_argument("--tracker", choices=["auto", "blob", "template", "csrt"], default="auto",
                    help="predict mode: auto (default) picks blob for a target darker than the background, else template;"
                         " csrt = OpenCV's, slower and loses the target on fast turret moves")
    ap.add_argument("--lead", type=float, default=0.03, help="predict mode: seconds to aim ahead (sim best: 0.03)")
    ap.add_argument("--q", type=float, default=2e6, help="predict mode: how fast the target can change direction")
    ap.add_argument("--latency", type=float, default=0.025, help="predict mode: camera delay, seconds (capture -> frame here)")
    ap.add_argument("--acc", type=int, help="servo ACC register 0-254 (default 60; 254 in predict mode)")
    a = ap.parse_args()
    if a.mode == "predict" and not a.port:
        sys.exit("predict mode reads the servo angles, so it needs --port (e.g. --port COM9).")
    global LIMITS
    LIMITS = load_limits()
    print(f"limits: pan {LIMITS[PAN_ID][0]:+.0f}..{LIMITS[PAN_ID][1]:+.0f} deg, tilt {LIMITS[TILT_ID][0]:+.0f}..{LIMITS[TILT_ID][1]:+.0f} deg")

    cap = open_camera(a.cam, a.width, a.height, a.fps, a.exposure)
    servos = Servos(a.port, a.acc if a.acc is not None else (254 if a.mode == "predict" else 60)) if a.port else None
    laser = Laser(a.nano) if a.nano else None
    os.makedirs("recordings", exist_ok=True)

    tracker, tname, box = None, "", None
    aim = None                                       # predict mode's Predictive
    angle_hist = deque(maxlen=64)                    # (time, pan, tilt) as read from the servos
    lens = None
    pan = tilt = 0.0; pan_sign, tilt_sign = -1.0, -1.0       # pan servo is flipped in the v2 turret
    mode, armed, writer, exposure = a.mode, False, None, a.exposure
    t_prev, fps_est = time.time(), 0.0
    win = "turret"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if servos: servos.goto(0, 0)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera stopped delivering frames"); break
            t_frame = time.perf_counter()
            raw = frame.copy()                       # clean copy: the target is picked from this, not the overlay
            H, W = frame.shape[:2]
            vfov = a.hfov * H / W
            if lens is None or (lens.w, lens.h) != (W, H):
                from turret_predict import Lens
                lens = Lens(W, H, hfov=a.hfov, diag_fov=a.dfov)
            if mode == "predict" and servos:
                r = servos.read_deg()
                if r is not None:                                 # store in the predictor's convention (+ = left / up)
                    angle_hist.append((time.perf_counter(), -pan_sign * r[0], -tilt_sign * r[1]))
            if writer is not None: writer.write(frame)
            now = time.time(); fps_est = 0.9 * fps_est + 0.1 / max(1e-3, now - t_prev); t_prev = now

            locked = False
            if mode == "predict" and aim is not None and angle_hist:
                t_cap = t_frame - a.latency
                pm, tm = angle_at(angle_hist, t_cap)               # where the turret pointed when the frame was taken
                goal, locked = aim.update(frame, t_cap, pm, tm, time.perf_counter())
                if goal is None:
                    aim = None
                    cv2.putText(frame, "TARGET LOST - press C", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    pan = float(np.clip(-pan_sign * goal[0], *LIMITS[PAN_ID]))
                    tilt = float(np.clip(-tilt_sign * goal[1], *LIMITS[TILT_ID]))
                    servos.goto(pan, tilt)
                    if aim.box is not None:
                        b = [int(v) for v in aim.box]
                        cv2.rectangle(frame, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (0, 255, 0) if locked else (0, 200, 255), 2)
                    u = __import__("turret_predict").world_to_pixel(lens, pm, tm, *aim.predicted(time.perf_counter() + a.lead))
                    if u is not None: cv2.circle(frame, (int(u[0]), int(u[1])), 6, (255, 0, 255), 2)   # where it's aiming
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
                        locked = abs(ex) < 2.0 and abs(ey) < 2.0
                    pan = float(np.clip(pan, *LIMITS[PAN_ID])); tilt = float(np.clip(tilt, *LIMITS[TILT_ID]))
                    if servos: servos.goto(pan, tilt)
                    cv2.rectangle(frame, box[:2], (box[0] + box[2], box[1] + box[3]), (0, 255, 0) if locked else (0, 200, 255), 2)
                else:
                    cv2.putText(frame, "TARGET LOST - press C", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if laser: laser.set(armed and locked)
            if tracker is None:
                cv2.putText(frame, "Press C (with this window selected) to pick a target", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # overlay (not recorded)
            cv2.drawMarker(frame, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 30, 1)
            hud = f"{mode}  {fps_est:4.0f} fps  pan {pan:+6.1f}  tilt {tilt:+5.1f}  exp {exposure:g}  {tname}"
            cv2.putText(frame, hud, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if writer is not None: cv2.circle(frame, (W - 30, 30), 10, (0, 0, 255), -1)
            if laser: cv2.putText(frame, "LASER " + ("ON" if armed and locked else "ARMED" if armed else "SAFE"),
                                  (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if armed else (200, 200, 200), 2)
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27): break
            elif k == ord("c"):
                r = cv2.selectROI(win, frame, showCrosshair=True)
                if r[2] > 4 and r[3] > 4:
                    if mode == "predict":
                        from turret_predict import Predictive
                        if angle_hist:
                            _, pm, tm = angle_hist[-1]
                            aim = Predictive(lens, a.tracker); aim.start(raw, r, pm, tm); tname = aim.name
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
        if servos: servos.goto(0, 0); time.sleep(0.5); servos.release()
        cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
