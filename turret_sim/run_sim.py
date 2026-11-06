"""Run every flight path against every controller and score how well the turret stays on the Tello.

    python run_sim.py                 table of results (+ out/results.csv, out/errors.png)
    python run_sim.py --video         also writes out/<scenario>_<controller>.mp4 (what the camera saw)
    python run_sim.py --only zigzag   one flight path

Timing model (per frame): capture -> camera latency -> tracker runs (its real compute time) -> goal sent to the servos.
If the tracker is still busy when new frames arrive, it takes the newest one (frames get dropped, like on the PC).
"""
import argparse, csv, math, os, sys, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim_world import Camera, Servo, make_room, scenarios, render, project, bearing, rot
from controllers import Baseline, Predictive

OUT = os.path.join(HERE, "out")


def ang_err(pan, tilt, p):
    a = rot(pan, tilt)[:, 0]; d = p / np.linalg.norm(p)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ d)))))


def run(path, ctrl, cam, servo_kw, duration=8.0, seed=0, room=None, video=None):
    rng = np.random.default_rng(seed)
    room = make_room(seed) if room is None else room
    dt = 0.001
    n = int(duration / dt) + 1
    p0 = path(0.0); b0 = bearing(p0)
    sp, st = Servo(**servo_kw), Servo(**servo_kw)
    sp.pos = sp.motor = sp.goal = b0[0]; st.pos = st.motor = st.goal = b0[1]
    hist = np.zeros((n, 2))
    frame0 = render(cam, room, b0[0], b0[1], [p0], rng)
    u, v, r = project(cam, b0[0], b0[1], p0)
    ctrl.start(frame0, (u - r, v - r * 0.6, 2 * r, 1.2 * r), b0[0], b0[1])
    period = 1.0 / cam.fps
    last_k, free_at, pending = 0, 0.0, []
    locked = False
    log = {"err": np.zeros(n), "hit": np.zeros(n, bool), "fire": np.zeros(n, bool), "proc": [], "lost": False}
    writer = None
    if video:
        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (cam.w, cam.h))
    vid_next = 0.0
    for i in range(n):
        t = i * dt
        while pending and pending[0][0] <= t:
            _, g = pending.pop(0)
            sp.goal = max(sp.limits[0], min(sp.limits[1], g[0])); st.goal = max(st.limits[0], min(st.limits[1], g[1]))
        sp.step(dt); st.step(dt)
        hist[i] = sp.pos, st.pos
        p = path(t)
        e = ang_err(sp.pos, st.pos, p)
        log["err"][i] = e
        log["hit"][i] = e < math.degrees(math.atan(0.09 / np.linalg.norm(p)))   # aim inside the drone's outline
        log["fire"][i] = locked
        if t + 1e-9 < free_at or log["lost"]:
            continue
        k = int((t - cam.latency) / period)
        if k <= last_k:
            continue
        last_k = k
        tc = k * period
        ic = int(round(tc / dt))
        pc, tcl = hist[ic]
        pts = [path(tc - cam.exposure * j / 2) for j in range(3)]
        frame = render(cam, room, pc, tcl, pts, rng)
        w0 = time.perf_counter()
        goal, locked = ctrl.update(frame, tc, pc + rng.normal(0, 0.05), tcl + rng.normal(0, 0.05), t)
        proc = min(0.05, max(0.002, time.perf_counter() - w0))
        log["proc"].append(proc)
        free_at = t + proc
        if goal is None and not locked and getattr(ctrl, "misses", 0) > 30:
            log["lost"] = True
        if goal is not None:
            pending.append((t + proc + 0.001, goal))
        if writer is not None and tc >= vid_next:
            vid_next += 1 / 30
            img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(img, (cam.w // 2, cam.h // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
            if ctrl.box is not None:
                b = [int(x) for x in ctrl.box]
                cv2.rectangle(img, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (0, 255, 0) if locked else (0, 200, 255), 1)
            cv2.putText(img, f"{ctrl.name[:38]}  t={tc:4.2f}s  err {e:4.1f} deg", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            writer.write(img)
    if writer is not None:
        writer.release()
    return log


def score(log, t0=0.5, dt=0.001):
    i0 = int(t0 / dt)
    err, hit, fire = log["err"][i0:], log["hit"][i0:], log["fire"][i0:]
    fire_ok = float(np.mean(hit[fire])) if fire.any() else float("nan")
    return {"on_target_%": 100 * float(np.mean(hit)), "mean_err_deg": float(np.mean(err)), "p95_err_deg": float(np.percentile(err, 95)),
            "laser_on_%": 100 * float(np.mean(fire)), "laser_hits_%": 100 * fire_ok, "lost": log["lost"],
            "track_ms": 1000 * float(np.mean(log["proc"])) if log["proc"] else float("nan")}


def controllers(cam, which):
    out = {"current": lambda: Baseline(cam),
           "kalman_csrt": lambda: Predictive(cam, "CSRT"),
           "kalman_auto": lambda: Predictive(cam, "AUTO", name="predictive (auto tracker + Kalman, lead 30 ms)"),
           "kalman_tmpl": lambda: Predictive(cam, "TEMPLATE", name="predictive (template + Kalman, lead 30 ms)"),
           "kalman_blob": lambda: Predictive(cam, "BLOB", name="predictive (dark-blob + Kalman, lead 30 ms)")}
    return {k: v for k, v in out.items() if not which or k in which}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*"); ap.add_argument("--ctrl", nargs="*")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--acc", type=float, default=2230.0, help="servo acceleration, deg/s^2 (ACC 60 = 527, ACC 254 = 2230)")
    ap.add_argument("--vmax", type=float, default=220.0, help="servo top speed, deg/s (SPEED 2500 = 220)")
    ap.add_argument("--duration", type=float, default=8.0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cam = Camera()
    print(f"camera: {cam.w}x{cam.h} @ {cam.fps} fps, HFOV {cam.hfov:.0f} deg, latency {cam.latency * 1000:.0f} ms;"
          f" servos {a.vmax:.0f} deg/s, {a.acc:.0f} deg/s^2")
    room = make_room(0)
    rows = []
    for sname, path in scenarios().items():
        if a.only and sname not in a.only: continue
        for cname, make in controllers(cam, a.ctrl).items():
            vid = os.path.join(OUT, f"{sname}_{cname}.mp4") if a.video else None
            log = run(path, make(), cam, dict(vmax=a.vmax, amax=a.acc), a.duration, room=room, video=vid)
            s = score(log); s.update(scenario=sname, controller=cname); rows.append(s)
            print(f"{sname:11s} {cname:12s} on target {s['on_target_%']:5.1f}%  mean {s['mean_err_deg']:5.2f} deg  p95 {s['p95_err_deg']:5.2f}"
                  f"  laser on {s['laser_on_%']:5.1f}% (hits {s['laser_hits_%']:5.1f}%)  track {s['track_ms']:4.1f} ms{'  LOST' if s['lost'] else ''}")
            np.save(os.path.join(OUT, f"err_{sname}_{cname}.npy"), log["err"])
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
