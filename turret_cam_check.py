"""First-day check for the turret camera (Arducam OV9281 or any UVC webcam).

    python turret_cam_check.py --cam 1            which sizes / frame rates the camera really delivers
    python turret_cam_check.py --cam 1 --view     live picture with the measured fps (Q to quit)
    python turret_cam_check.py --cam 1 --latency  measures the camera delay to use as turret_track.py --latency

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


def open_cam(idx, w, h, fps, mjpg=True):
    from camopen import open_capture
    cap, _ = open_capture(idx, quiet=True)
    if cap is None: return None
    if mjpg: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h); cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def measure(cap, n=120):
    for _ in range(10): cap.read()                       # let it settle
    t0 = time.perf_counter(); got = 0
    for _ in range(n):
        ok, f = cap.read()
        if ok: got += 1
    dt = time.perf_counter() - t0
    return got / dt if dt > 0 else 0.0, f


def probe(idx):
    print(f"camera {idx}: asking for each mode and timing {120} frames ...")
    best = None
    for w, h, fps in MODES:
        cap = open_cam(idx, w, h, fps)
        if cap is None: sys.exit(f"camera {idx} didn't open. Try --cam 0/1/2 and close other apps using it.")
        rw, rh = int(cap.get(3)), int(cap.get(4))
        real, f = measure(cap)
        mono = f is not None and f.ndim == 3 and np.allclose(f[..., 0], f[..., 1], atol=2)
        print(f"  asked {w}x{h}@{fps:<3}  got {rw}x{rh}  measured {real:5.1f} fps{'  (mono)' if mono else ''}")
        if (rw, rh) == (w, h) and (best is None or real > best[2] * 1.05): best = (w, h, real)
        cap.release()
    if best:
        w, h, r = best
        print(f"\nfastest: {w}x{h} at {r:.0f} fps.  Use: turret_track.py --cam {idx} --width {w} --height {h} --fps {round(r / 10) * 10} --dfov 130")


def view(idx, w, h, fps):
    cap = open_cam(idx, w, h, fps)
    t_prev, est = time.perf_counter(), 0.0
    while True:
        ok, f = cap.read()
        if not ok: break
        t = time.perf_counter(); est = 0.9 * est + 0.1 / max(1e-4, t - t_prev); t_prev = t
        cv2.putText(f, f"{f.shape[1]}x{f.shape[0]}  {est:5.1f} fps   Q quit", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("camera", f)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
    cap.release(); cv2.destroyAllWindows()


def latency(idx, w, h, fps, flashes=20):
    cap = open_cam(idx, w, h, fps)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640); ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--fps", type=int, default=120)
    ap.add_argument("--view", action="store_true"); ap.add_argument("--latency", action="store_true")
    a = ap.parse_args()
    if a.view: view(a.cam, a.width, a.height, a.fps)
    elif a.latency: latency(a.cam, a.width, a.height, a.fps)
    else: probe(a.cam)


if __name__ == "__main__":
    main()
