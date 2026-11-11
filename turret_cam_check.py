"""First-day check for the turret camera (Arducam OV9281 or any UVC webcam).

    python turret_cam_check.py --cam 1            which sizes / frame rates the camera really delivers
    python turret_cam_check.py --cam 1 --view     live picture with the measured fps (Q to quit)
    python turret_cam_check.py --cam 1 --latency  measures the camera delay to use as turret_track.py --latency
    python turret_cam_check.py --cam 0 --laser-latency --nano COM11   same, timed with the turret's laser (better:
                                                  no monitor delay in it). Point the turret at a wall, not at anyone.
    python turret_cam_check.py --cam 0 --width 1280 --height 720 --exposure -6 --laser-cal --nano COM11
                                                  where does the laser land vs the camera's centre? Measure at 2-3
                                                  distances from a wall; prints the turret_track.py correction flags.

Latency test: a window flashes black/white. Point the camera at it (fill most of the picture, room lights dim is
fine) and keep still. It times each flash from the moment it's drawn to the moment the camera sees it, 20 times.
That includes the monitor's own delay (about 5-15 ms on a 60-144 Hz screen), so the number is a little high:
subtract ~8 ms if your monitor is 60 Hz, ~4 ms at 144 Hz.
"""
import argparse, os, sys, time
import numpy as np
import cv2

MODES = [(1280, 800, 120), (1280, 720, 120), (640, 400, 120), (640, 480, 120), (1280, 800, 60), (640, 400, 210),
         (1280, 720, 60), (640, 480, 60)]


EXPOSURE = None                          # set from --exposure in main()


def open_cam(idx, w, h, fps, mjpg=True):
    from camopen import open_capture
    cap, backend = open_capture(idx, w, h, fps, mjpg, quiet=True)
    if cap is None: return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if EXPOSURE is not None:
        # fixed short exposure: on auto, a dim room makes the camera slow its frame rate down (and add delay)
        from camopen import set_exposure
        if not set_exposure(cap, EXPOSURE): print(f"(the camera ignored --exposure {EXPOSURE} on {backend})")
    return cap


def measure(cap, n=120):
    f = None
    try:
        for _ in range(10): cap.read()                   # let it settle
    except cv2.error:
        return 0.0, None
    t0 = time.perf_counter(); got = 0
    for _ in range(n):
        try:
            ok, fr = cap.read()
        except cv2.error:
            ok = False
        if ok: got += 1; f = fr
    dt = time.perf_counter() - t0
    return got / dt if dt > 0 else 0.0, f


def probe(idx):
    print(f"camera {idx}: asking for each mode and timing {120} frames ...")
    best = None
    for w, h, fps in MODES:
        cap = open_cam(idx, w, h, fps)
        if cap is None:
            print(f"  asked {w}x{h}@{fps:<3}  could not open in this mode"); continue
        rw, rh = int(cap.get(3)), int(cap.get(4))
        real, f = measure(cap)
        mono = f is not None and f.ndim == 3 and np.allclose(f[..., 0], f[..., 1], atol=2)
        from camopen import fourcc
        print(f"  asked {w}x{h}@{fps:<3}  got {rw}x{rh} {fourcc(cap)}  measured {real:5.1f} fps{'  (mono)' if mono else ''}"
              f"  [{cap.getBackendName()}]")
        if (rw, rh) == (w, h) and (best is None or real > best[2] * 1.05): best = (w, h, real)
        cap.release()
    if best:
        w, h, r = best
        print(f"\nfastest: {w}x{h} at {r:.0f} fps.  Use: turret_track.py --cam {idx} --width {w} --height {h} --fps {round(r / 10) * 10} --dfov 130")


def need(cap, idx):
    if cap is None:
        sys.exit(f"camera {idx} didn't open. Close the Windows Camera app (and anything else using it), "
                 "unplug and replug the camera, wait 5 s, and try again.")
    return cap


def view(idx, w, h, fps):
    cap = need(open_cam(idx, w, h, fps), idx)
    t_prev, est = time.perf_counter(), 0.0
    while True:
        ok, f = cap.read()
        if not ok: break
        t = time.perf_counter(); est = 0.9 * est + 0.1 / max(1e-4, t - t_prev); t_prev = t
        cv2.putText(f, f"{f.shape[1]}x{f.shape[0]}  {est:5.1f} fps  {cap.getBackendName()}   Q quit", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("camera", f)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
    cap.release(); cv2.destroyAllWindows()


def latency(idx, w, h, fps, flashes=20):
    cap = need(open_cam(idx, w, h, fps), idx)
    flash = "flash (point the camera here)"
    cv2.namedWindow(flash, cv2.WINDOW_NORMAL); cv2.resizeWindow(flash, 900, 700)
    black = np.zeros((700, 900, 3), np.uint8); white = np.full_like(black, 255)
    # aim phase: show white with the camera view inset until SPACE
    print("Aim the camera at the flash window so it fills most of the picture, then press SPACE.")
    while True:
        ok, f = cap.read()
        if not ok: sys.exit("camera stopped")
        img = white.copy()
        small = cv2.resize(f, (300, int(300 * f.shape[0] / f.shape[1])))
        img[10:10 + small.shape[0], 10:310] = small
        cv2.putText(img, "SPACE = start   Q = quit", (330, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imshow(flash, img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "): break
        if k in (ord("q"), 27): return
    lo = hi = None; results = []
    state = False
    cv2.imshow(flash, black); cv2.waitKey(1)
    for _ in range(60): cap.read()
    lo = np.mean([cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY).mean() for _ in range(10)])
    cv2.imshow(flash, white); cv2.waitKey(1)
    for _ in range(60): cap.read()
    hi = np.mean([cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY).mean() for _ in range(10)])
    if hi - lo < 20: sys.exit(f"the camera barely sees the flash (dark {lo:.0f}, bright {hi:.0f}); move it closer")
    mid = (lo + hi) / 2; state = True
    for i in range(flashes):
        state = not state
        cv2.imshow(flash, white if state else black); cv2.waitKey(1)
        t0 = time.perf_counter()
        while True:
            ok, f = cap.read()
            m = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()
            if (m > mid) == state:
                results.append(1000 * (time.perf_counter() - t0)); break
            if time.perf_counter() - t0 > 1.0: break
        time.sleep(0.25 + 0.1 * np.random.rand())
    cap.release(); cv2.destroyAllWindows()
    if not results: sys.exit("no flashes detected")
    r = np.array(results)
    print(f"camera + screen delay over {len(r)} flashes: median {np.median(r):.0f} ms (min {r.min():.0f}, max {r.max():.0f})")
    est = max(5.0, np.median(r) - 8.0)
    print(f"use about: turret_track.py --latency {est / 1000:.3f}   (after taking ~8 ms off for a 60 Hz monitor)")


def laser_latency(idx, w, h, fps, port, shots=20):
    """Switch the laser on/off through the ESP32 and time how long until the camera sees the dot change.
    Only the serial write (~1 ms) and the camera are in this number, so it's what --latency should be."""
    import serial
    cap = need(open_cam(idx, w, h, fps), idx)
    s = serial.Serial(port, 115200, timeout=0); time.sleep(2.0)
    gray = lambda: cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY).astype(np.int16)
    def settle(sec):
        t = time.perf_counter()
        while time.perf_counter() - t < sec: cap.read()
    s.write(b"0"); settle(0.6)
    off = gray()
    s.write(b"1"); settle(0.3)
    on = gray(); s.write(b"0")
    diff = cv2.GaussianBlur(np.abs(on - off).astype(np.uint8), (0, 0), 2)
    _, peak, _, (px, py) = cv2.minMaxLoc(diff)
    if peak < 25:
        s.close(); sys.exit(f"can't see the laser dot (change {peak:.0f}). Is the laser armed/plugged in, and pointing "
                            "at a wall inside the camera's view? Dim the room lights if it's bright.")
    r = 12
    roi = lambda g: float(g[max(0, py - r):py + r, max(0, px - r):px + r].mean())
    lo, hi = roi(off), roi(on); mid = (lo + hi) / 2
    n0, t0f = 0, time.perf_counter()
    while time.perf_counter() - t0f < 1.0: cap.read(); n0 += 1
    from camopen import fourcc
    print(f"backend {cap.getBackendName()}, format {fourcc(cap)}, camera running at {n0:.0f} fps during the test")
    print(f"laser dot at pixel ({px}, {py}); timing {shots} on/off switches ...")
    res = []
    settle(0.6)
    for i in range(shots):
        want_on = i % 2 == 0
        t0 = time.perf_counter(); s.write(b"1" if want_on else b"0")
        while time.perf_counter() - t0 < 0.45:            # the ESP32 turns the laser off by itself after 0.5 s
            g = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY).astype(np.int16)
            if (roi(g) > mid) == want_on:
                res.append(1000 * (time.perf_counter() - t0)); break
        s.write(b"0"); settle(0.3 + 0.1 * np.random.rand())
    s.write(b"0"); s.close(); cap.release()
    if not res: sys.exit("no switches detected")
    r_ = np.array(res)
    print(f"camera delay over {len(r_)} switches: median {np.median(r_):.0f} ms (min {r_.min():.0f}, max {r_.max():.0f})")
    print(f"use: turret_track.py --latency {np.median(r_) / 1000 - 0.004:.3f}   (half a frame taken off: the frame is exposed before it arrives)")


def laser_cal(idx, w, h, fps, port, focal=641.0):
    """Laser dot vs camera centre at several distances from a wall -> the laser's offset from the lens (metres,
    matters up close) and its angle error (degrees, the same at every distance). turret_track.py aims with both."""
    import serial
    from turret_predict import Lens, pixel_bearing
    cap = need(open_cam(idx, w, h, fps), idx)
    lens = Lens(w, h, focal=focal * w / 1280.0, fisheye=True)
    s = serial.Serial(port, 115200, timeout=0); time.sleep(2.0)
    last = [0.0]

    def laser(on):                                        # the ESP32 turns the laser off after 0.5 s without a '1'
        if time.perf_counter() - last[0] > 0.1 or not on:
            s.write(b"1" if on else b"0"); last[0] = time.perf_counter()

    print("Point the turret square at a wall (nobody in front of it). For each distance:\n"
          "  SPACE   freeze the picture (laser on)\n"
          "  click   on the dot (it snaps to the brightest spot nearby; a magnifier follows the mouse); click again to redo\n"
          "  type    the distance from the LENS to the DOT in metres, then ENTER   (ESC = cancel this one)\n"
          "Do 2-3 distances, close and far (e.g. 0.5, 1.5, 3+). Q when done.")
    pts = []
    win = "laser cal"; cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    mouse = {"xy": None, "click": None}

    def on_mouse(ev, x, y, flags, _):
        mouse["xy"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN: mouse["click"] = (x, y)
    cv2.setMouseCallback(win, on_mouse)

    def snap(img, x, y, r=12):
        """Brightest spot within r px of the click (centre of brightness, sub-pixel)."""
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
        x0, y0 = max(0, x - r), max(0, y - r)
        roi = g[y0:y + r + 1, x0:x + r + 1]
        if roi.size == 0: return float(x), float(y)
        pk = roi.max(); lo = np.median(roi)
        if pk - lo < 10: return float(x), float(y)                   # nothing bright there: take the click as is
        wgt = np.where(roi > lo + 0.6 * (pk - lo), roi - lo, 0)
        ys, xs = np.mgrid[y0:y0 + roi.shape[0], x0:x0 + roi.shape[1]]
        return float((xs * wgt).sum() / wgt.sum()), float((ys * wgt).sum() / wgt.sum())

    def draw(img, dot=None, typed=None, msg=""):
        v = img.copy(); cx, cy = int(lens.cx), int(lens.cy)
        cv2.line(v, (cx - 25, cy), (cx + 25, cy), (0, 255, 255), 1); cv2.line(v, (cx, cy - 25), (cx, cy + 25), (0, 255, 255), 1)
        for p in pts: cv2.circle(v, (int(p[1][0]), int(p[1][1])), 8, (0, 255, 0), 1)
        if dot: cv2.drawMarker(v, (int(round(dot[0])), int(round(dot[1]))), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        if mouse["xy"]:                                           # 6x magnifier of the area under the mouse
            mx, my = mouse["xy"]; r = 20
            crop = img[max(0, my - r):my + r, max(0, mx - r):mx + r]
            if crop.shape[0] > 4 and crop.shape[1] > 4:
                z = cv2.resize(crop, (crop.shape[1] * 6, crop.shape[0] * 6), interpolation=cv2.INTER_NEAREST)
                zh, zw = z.shape[:2]; v[10:10 + zh, v.shape[1] - 10 - zw:v.shape[1] - 10] = z
                cv2.rectangle(v, (v.shape[1] - 10 - zw, 10), (v.shape[1] - 10, 10 + zh), (0, 255, 255), 1)
        line = msg or (f"{len(pts)} measured   SPACE freeze   Q done" if typed is None else f"distance (m): {typed}_   ENTER ok   ESC cancel")
        cv2.putText(v, line, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return v

    while True:
        laser(True)
        ok, fr = cap.read()
        if not ok: continue
        cv2.imshow(win, draw(fr))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        if k != ord(" "): continue
        still = fr.copy(); laser(False)                          # frozen picture; laser off while you click
        dot, typed, mouse["click"] = None, None, None
        while True:
            if mouse["click"]:
                dot = snap(still, *mouse["click"]); mouse["click"] = None
                if typed is None: typed = ""
            cv2.imshow(win, draw(still, dot, typed, "" if dot else "click on the laser dot   ESC cancel"))
            k = cv2.waitKey(20) & 0xFF
            if k == 27: dot = None; break
            if typed is not None:
                if k in (13, 10) and typed: break
                if k == 8: typed = typed[:-1]
                elif chr(k) in "0123456789." and k != 255: typed += chr(k)
        if dot is None: print("  cancelled"); continue
        try:
            d = float(typed)
        except ValueError:
            print("  not a number, skipped"); continue
        bp, bt = pixel_bearing(lens, 0.0, 0.0, *dot)
        right, down = -bp, -bt                             # bearings are + left / + up
        print(f"  {d:.2f} m: dot at pixel ({dot[0]:.1f}, {dot[1]:.1f}) = {right:+.2f} deg right, {down:+.2f} deg down")
        pts.append((d, dot, right, down))
    s.write(b"0"); s.close(); cap.release(); cv2.destroyAllWindows()
    if len(pts) < 2:
        sys.exit("Need at least 2 distances (one close, one far) to separate the laser's offset from its angle.")
    d = np.array([p[0] for p in pts]); R = np.array([p[2] for p in pts]); Dn = np.array([p[3] for p in pts])
    # dot angle at distance d = atan(offset / d) + trim  ~=  offset * (1/d) [rad] + trim
    A = np.stack([np.degrees(1.0 / d), np.ones_like(d)], 1)
    (mr, tr), *_ = np.linalg.lstsq(A, R, rcond=None)
    (mb, tb), *_ = np.linalg.lstsq(A, Dn, rcond=None)
    print("\n dist   right   down   (deg; fit error)")
    for i in range(len(d)):
        fr_, fd_ = A[i] @ [mr, tr], A[i] @ [mb, tb]
        print(f" {d[i]:4.2f}m  {R[i]:+6.2f}  {Dn[i]:+6.2f}   ({R[i] - fr_:+.2f}, {Dn[i] - fd_:+.2f})")
    print(f"\nlaser sits {mb * 100:+.1f} cm below and {mr * 100:+.1f} cm right of the lens, and points "
          f"{tb:+.2f} deg down, {tr:+.2f} deg right of the camera's centre")
    print(f"use:  turret_track.py ... --muzzle-below {mb:.3f} --muzzle-right {mr:.3f} "
          f"--laser-trim-down {tb:.2f} --laser-trim-right {tr:.2f}")


def diagnose(idx, w, h, fps):
    """Try every known way of opening the camera in this mode and report format + real frame rate for each."""
    from camopen import fourcc
    M = cv2.VideoWriter_fourcc(*"MJPG")
    F, W_, H_, R = cv2.CAP_PROP_FOURCC, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS
    recipes = [
        ("DSHOW  fourcc,size,fps", cv2.CAP_DSHOW, None, [(F, M), (W_, w), (H_, h), (R, fps)]),
        ("DSHOW  size,fourcc,fps", cv2.CAP_DSHOW, None, [(W_, w), (H_, h), (F, M), (R, fps)]),
        ("DSHOW  fps,fourcc,size", cv2.CAP_DSHOW, None, [(R, fps), (F, M), (W_, w), (H_, h)]),
        ("DSHOW  open-params", cv2.CAP_DSHOW, [F, M, W_, w, H_, h, R, fps], []),
        ("MSMF   open-params size,fps", cv2.CAP_MSMF, [W_, w, H_, h, R, fps], []),
        ("MSMF   open-params size", cv2.CAP_MSMF, [W_, w, H_, h], []),
        ("MSMF   plain, set size", cv2.CAP_MSMF, None, [(W_, w), (H_, h)]),
    ]
    print(f"camera {idx}, asking for {w}x{h} @ {fps}. Keep the room lights on. Each recipe takes a few seconds.\n")
    for label, api, params, sets in recipes:
        try:
            cap = cv2.VideoCapture(idx, api, params) if params else cv2.VideoCapture(idx, api)
        except cv2.error as e:
            print(f"  {label:30s} open error"); time.sleep(2); continue
        if not cap.isOpened():
            print(f"  {label:30s} did not open"); time.sleep(2); continue
        for prop, val in sets:
            cap.set(prop, val)
        got, t0, ok_any = 0, None, False
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < 4.0:
            try:
                ok, f = cap.read()
            except cv2.error:
                ok = False
            if ok:
                ok_any = True
                if t0 is None: t0 = time.perf_counter()
                else: got += 1
                if time.perf_counter() - t0 > 1.5: break
        rate = got / (time.perf_counter() - t0) if t0 and got else 0.0
        size = f"{int(cap.get(3))}x{int(cap.get(4))}"
        print(f"  {label:30s} {'frames' if ok_any else 'NO FRAMES':9s} {size:>9s} {fourcc(cap):5s} {rate:6.1f} fps")
        cap.release(); time.sleep(2.0)


def measure_fov(idx, w, h, fps, port, step=8.0):
    """Turn the turret a known amount and see how far the picture moves: gives the real field of view (and checks
    which way each servo turns). Point it at a scene with some detail, not a blank wall, and keep things still."""
    import math
    from turret_track import Servos
    cap = need(open_cam(idx, w, h, fps), idx)
    servos = Servos(port, acc=60)

    def still(pan, tilt):
        servos.goto(pan, tilt)
        t = time.perf_counter()
        while time.perf_counter() - t < 1.2: cap.read()            # let it get there and stop shaking
        fr = [cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY).astype(np.float32) for _ in range(3)]
        return np.mean(fr, axis=0)

    win = None
    def shift(a, b):
        nonlocal win
        if win is None: win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
        return dx, dy, resp

    print(f"turning the turret {step:g} deg each way and measuring how far the picture moves ...")
    base = still(0, 0)
    fs = {}
    for axis, (p, t) in (("pan", (step, 0)), ("tilt", (0, step))):
        moved = still(p, t)
        dx, dy, resp = shift(base, moved)
        px = dx if axis == "pan" else dy
        f = abs(px) / math.tan(math.radians(step))
        fs[axis] = f
        way = ("right" if dx < 0 else "left") if axis == "pan" else ("up" if dy > 0 else "down")
        print(f"  {axis}: servo +{step:g} deg moved the picture {px:+.0f} px  -> focal length {f:.0f} px; "
              f"servo + turns the camera {way}  (match quality {resp:.2f})")
        base = still(0, 0)
    servos.goto(0, 0); time.sleep(0.8); servos.release(); cap.release()
    f = (fs["pan"] + fs["tilt"]) / 2
    W, H = int(cap.get(3)) or w, int(cap.get(4)) or h
    hfov = math.degrees(2 * math.atan(w / 2 / f))
    print(f"\nfocal length about {f:.0f} px at {w}x{h}: horizontal field of view {hfov:.0f} deg")
    print(f"use: turret_track.py ... --width {w} --height {h} --hfov {hfov:.0f}")
    if min(fs.values()) > 0 and abs(fs["pan"] - fs["tilt"]) / f > 0.15:
        print("(pan and tilt disagree by more than 15%: something in view moved, or the scene has too little detail. Run it again.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640); ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--fps", type=int, default=120)
    ap.add_argument("--view", action="store_true"); ap.add_argument("--latency", action="store_true")
    ap.add_argument("--laser-latency", action="store_true"); ap.add_argument("--nano", default="COM11")
    ap.add_argument("--laser-cal", action="store_true", help="measure where the laser lands vs the camera centre (needs --nano)")
    ap.add_argument("--focal", type=float, default=641.0, help="fisheye focal length at 1280 px wide (--laser-cal)")
    ap.add_argument("--diagnose", action="store_true", help="try every way of opening the camera; report fps for each")
    ap.add_argument("--fov", action="store_true", help="measure the real field of view by turning the turret (needs --port)")
    ap.add_argument("--port", default="COM9", help="turret servo board, for --fov")
    ap.add_argument("--exposure", type=float, help="fixed exposure, DirectShow log2 seconds (-8 = 1/256 s, -9 = 1/512 s); default auto")
    a = ap.parse_args()
    global EXPOSURE; EXPOSURE = a.exposure
    if a.diagnose: diagnose(a.cam, a.width, a.height, a.fps)
    elif a.fov: measure_fov(a.cam, a.width, a.height, a.fps, a.port)
    elif a.view: view(a.cam, a.width, a.height, a.fps)
    elif a.latency: latency(a.cam, a.width, a.height, a.fps)
    elif a.laser_latency: laser_latency(a.cam, a.width, a.height, a.fps, a.nano)
    elif a.laser_cal: laser_cal(a.cam, a.width, a.height, a.fps, a.nano, a.focal)
    else: probe(a.cam)


if __name__ == "__main__":
    main()
