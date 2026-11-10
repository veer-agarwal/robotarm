"""Predictive aiming for the turret (used by turret_track.py --mode predict, and by turret_sim/).

The plain follow mode nudges the servos toward wherever the target sits in the picture. By the time a nudge lands,
the picture is ~40 ms old, so on a fast drone it always trails behind. This version:

  1. turns the target's pixel into a WORLD direction (turret angle when the frame was taken + exact lens maths),
  2. runs a constant-velocity Kalman filter on that direction (so it knows how fast the target is swinging),
  3. aims where the target will be once the command has taken effect (lead ~30 ms).

In the simulator (turret_sim/run_sim.py) this took a zig-zagging Tello from 26% to 99% of the time on target.

Angles: pan + = left (seen from behind the turret), tilt + = up, degrees.
"""
import math
import numpy as np
import cv2


# ---------------------------------------------------------------- geometry
def rot(pan, tilt):
    """World-from-camera rotation for turret angles in degrees (x forward, y left, z up)."""
    p, t = math.radians(pan), math.radians(tilt)
    Rz = np.array([[math.cos(p), -math.sin(p), 0], [math.sin(p), math.cos(p), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(t), 0, -math.sin(t)], [0, 1, 0], [math.sin(t), 0, math.cos(t)]])
    return Rz @ Ry


def bearing(p):
    """(pan, tilt) in degrees pointing along vector p."""
    return math.degrees(math.atan2(p[1], p[0])), math.degrees(math.atan2(p[2], math.hypot(p[0], p[1])))


class Lens:
    """w x h picture, focal length f in pixels.
    fisheye=False: pinhole (straight lines stay straight), angle = atan(r / f).
    fisheye=True:  equidistant fisheye, angle = r / f. Wide M12 lenses like the Arducam 130 deg are close to this:
                   accurate near the centre AND at the edges, where the pinhole model is off by up to ~10 deg.
    Give focal (px, e.g. from turret_cam_check --fov), or hfov / diag_fov."""

    def __init__(self, w, h, hfov=None, diag_fov=None, focal=None, fisheye=False):
        self.w, self.h, self.cx, self.cy = w, h, w / 2, h / 2
        self.fisheye = fisheye
        if focal is not None:
            self.f = focal
        elif fisheye:
            self.f = (w / 2) / math.radians(hfov / 2)
        elif diag_fov is not None:
            self.f = math.hypot(w, h) / 2 / math.tan(math.radians(diag_fov / 2))
        else:
            self.f = w / 2 / math.tan(math.radians(hfov / 2))


class Scaled:
    """The same lens, for a picture resized by k (big targets are tracked on a smaller copy of the frame)."""

    def __init__(self, cam, k):
        self.w, self.h = int(round(cam.w * k)), int(round(cam.h * k))
        self.f, self.cx, self.cy = cam.f * k, cam.cx * k, cam.cy * k
        self.fisheye = getattr(cam, "fisheye", False)


WARMUP = 6            # the first detections after C set the target's real size (the drawn box is only a guess)
COAST_DAMP = 0.75      # while the target is missing, its estimated speed shrinks by this much every frame
TRACK_PX = 40          # big targets get tracked on a picture shrunk so they're about this many pixels across


def pixel_bearing(cam, pan, tilt, px, py):
    """World (pan, tilt) of the ray through pixel (px, py) when the turret points at (pan, tilt)."""
    dx, dy = px - cam.cx, py - cam.cy
    if getattr(cam, "fisheye", False):
        r = math.hypot(dx, dy)
        th = r / cam.f
        s = math.sin(th) / r if r > 1e-9 else 1.0 / cam.f
        ray = np.array([math.cos(th), -dx * s, -dy * s])
    else:
        ray = np.array([cam.f, -dx, -dy])
    return bearing(rot(pan, tilt) @ ray)


def world_to_pixel(cam, pan, tilt, bp, bt):
    """Where world direction (bp, bt) lands in the picture when the turret points at (pan, tilt); None if behind."""
    c = rot(pan, tilt).T @ rot(bp, bt)[:, 0]
    if getattr(cam, "fisheye", False):
        side = math.hypot(c[1], c[2])
        th = math.atan2(side, c[0])
        if th > math.radians(100): return None
        if side < 1e-9: return cam.cx, cam.cy
        r = cam.f * th
        return cam.cx - r * c[1] / side, cam.cy - r * c[2] / side
    if c[0] <= 0.05: return None
    return cam.cx - cam.f * c[1] / c[0], cam.cy - cam.f * c[2] / c[0]


# ---------------------------------------------------------------- filter
class KF1:
    """Constant-velocity Kalman filter for one angle. q = how hard the target can change speed (bigger = snappier,
    noisier), r = measurement noise in deg^2."""

    def __init__(self, x0, q=2e6, r=0.25):
        self.x = np.array([x0, 0.0]); self.P = np.diag([1.0, 400.0]); self.q, self.r = q, r; self.t = None

    def predict_to(self, t):
        if self.t is None: self.t = t; return
        dt = max(0.0, t - self.t); self.t = t
        F = np.array([[1, dt], [0, 1]])
        Q = self.q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        self.x = F @ self.x; self.P = F @ self.P @ F.T + Q

    def correct(self, z):
        S = self.P[0, 0] + self.r; K = self.P[:, 0] / S
        self.x = self.x + K * (z - self.x[0]); self.P = self.P - np.outer(K, self.P[0])

    def at(self, t):
        return self.x[0] + self.x[1] * (t - self.t)


def make_cv_tracker(kind):
    f = getattr(cv2, f"Tracker{kind}_create", None) or getattr(getattr(cv2, "legacy", None), f"Tracker{kind}_create", None)
    if f is None: raise RuntimeError(f"OpenCV has no {kind} tracker. Run: pip install opencv-contrib-python")
    return f()


def bgr(frame):
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame


def gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def tighten(g, box, margin=35):
    """The box drawn with C is usually loose. If it holds one clearly dark shape, shrink the box to that shape."""
    x, y, w, h = [int(v) for v in box]
    x, y = max(0, x), max(0, y); w, h = min(w, g.shape[1] - x), min(h, g.shape[0] - y)
    roi = g[y:y + h, x:x + w]
    if roi.size < 16: return box
    thr = float(np.median(roi)) - margin
    n, lab, st, _ = cv2.connectedComponentsWithStats((roi < thr).astype(np.uint8), 8)
    if n < 2: return box
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh, area = st[i]
    if area < 0.05 * w * h or area < 12: return box
    return (x + int(bx), y + int(by), int(bw), int(bh))


def darker_than_surroundings(g, box, margin=40):
    """True if the inside of box is clearly darker than a ring around it (then the BLOB tracker works)."""
    x, y, w, h = [int(v) for v in box]; m = max(4, max(w, h))
    inner = g[max(0, y):y + h, max(0, x):x + w]
    outer = g[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
    if inner.size == 0 or outer.size <= inner.size: return False
    return float(np.percentile(inner, 30)) < float(np.median(outer)) - margin


# ---------------------------------------------------------------- the aimer
class Predictive:
    """tracker:
      "AUTO"      BLOB if the picked target is clearly darker than what's around it, else TEMPLATE. Default.
      "TEMPLATE"  any target (~1-2 ms). Remembers what the target looks like and looks for it only around where the
                  filter says it should be in this frame, so the turret's own fast moves can't outrun it. Weak spots:
                  big size changes (a close pass) and crossing something that looks the same.
      "BLOB"      a target darker than what's behind it (~2 ms). Same search window; learns how dark the target is;
                  falls back to the template when it crosses something just as dark.
      "CSRT"      OpenCV's tracker (~10-20 ms). Searches near its last box in the picture, so a fast slew loses it.

    update(frame, t_cap, pan_at_capture, tilt_at_capture, t_now) -> ((pan_goal, tilt_goal) or None, locked)"""

    def __init__(self, cam, tracker="AUTO", lead=0.03, q=2e6, give_up=30, gate=8.0, name=None):
        self.cam, self.kind, self.lead, self.q, self.give_up, self.gate = cam, tracker.upper(), lead, q, give_up, gate
        self.name = name or f"predictive ({self.kind} + Kalman, lead {lead * 1000:.0f} ms)"
        self.kp = self.kt = None; self.box = None; self.misses = 0

    def start(self, frame, box, pan, tilt):
        if self.kind in ("AUTO", "BLOB"):
            box = tighten(gray(frame), box)
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        bp, bt = pixel_bearing(self.cam, pan, tilt, cx, cy)
        self.kp, self.kt = KF1(bp, self.q), KF1(bt, self.q)
        self.size = max(box[2], box[3]); self.box = box; self.misses = 0; self.hits = 0
        self.tmpl = None; self.level = None; self.anchor = None
        if self.kind == "AUTO":
            self.kind = "BLOB" if darker_than_surroundings(gray(frame), box) else "TEMPLATE"
            self.name = self.name.replace("AUTO", f"auto: {self.kind.lower()}")
        self.need, self.need_anchor, self.blend = 0.55, 0.4, 0.05
        if self.kind in ("BLOB", "TEMPLATE"):
            # matching a 380-px template over a 1000-px window takes ~100 ms; shrink the picture for big targets
            self.k = min(1.0, TRACK_PX / max(1, max(box[2], box[3])))
            self.full_cam, self.small_cam = self.cam, Scaled(self.cam, self.k)
            self.size = self.size * self.k
            self._learn(self._small(frame), tuple(int(round(v * self.k)) for v in box))
        else:
            self.trk = make_cv_tracker(self.kind); self.trk.init(bgr(frame), tuple(int(v) for v in box))

    def size_px(self):
        """Smoothed target size in full-picture pixels (steadier than one frame's box)."""
        return self.size / getattr(self, "k", 1.0)

    def predicted(self, t):
        return self.kp.at(t), self.kt.at(t)

    # -- BLOB tracker
    def _learn(self, g, box, blend=1.0):
        x, y, w, h = [int(v) for v in box]; m = max(2, int(0.3 * max(w, h)))
        x, y = max(0, min(x, g.shape[1] - 2)), max(0, min(y, g.shape[0] - 2))
        w, h = max(2, min(w, g.shape[1] - x)), max(2, min(h, g.shape[0] - y))
        # near an edge the margin would fall off the picture: pad with the edge pixels instead of giving up
        gp = cv2.copyMakeBorder(g, m, m, m, m, cv2.BORDER_REPLICATE)
        t = gp[y:y + h + 2 * m, x:x + w + 2 * m].astype(np.float32)
        if t.shape[0] < 5 or t.shape[1] < 5: return
        if blend < 1.0 and self.tmpl is not None and self.tmpl[0].shape == t.shape:
            t = (1 - blend) * self.tmpl[0] + blend * t                  # drift slowly toward the new look
        self.tmpl = (t, m)
        px = g[max(0, y):y + h, max(0, x):x + w]
        if px.size:
            lv = float(np.percentile(px, 30))
            self.level = lv if self.level is None else 0.9 * self.level + 0.1 * lv

    def _window(self, g, pan, tilt, t_cap):
        """The patch of this frame where the filter expects the target (bigger while it's been missing)."""
        uv = world_to_pixel(self.cam, pan, tilt, *self.predicted(t_cap))
        if uv is None: return None
        u, v = uv
        half = int(max(24, 3 * self.size) * (1 + 0.5 * self.misses))
        x0, y0 = max(0, int(u) - half), max(0, int(v) - half)
        x1, y1 = min(self.cam.w, int(u) + half), min(self.cam.h, int(v) + half)
        if x1 - x0 < 8 or y1 - y0 < 8: return None
        return g[y0:y1, x0:x1].astype(np.float32), x0, y0, u, v

    def _match(self, win, x0, y0, need=0.7):
        if self.tmpl is None: return None
        t, m = self.tmpl
        if t.shape[0] >= win.shape[0] or t.shape[1] >= win.shape[1]: return None
        r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, (bx, by) = cv2.minMaxLoc(r)
        if score < need: return None
        return (bx + x0 + m, by + y0 + m, t.shape[1] - 2 * m, t.shape[0] - 2 * m)

    def _template(self, g, pan, tilt, t_cap):
        """Match the remembered look. It's updated slowly, and a match must still resemble the look picked with C,
        so it can't creep onto the background."""
        w = self._window(g, pan, tilt, t_cap)
        if w is None or self.tmpl is None: return None
        win, x0, y0 = w[0], w[1], w[2]
        t, m = self.tmpl
        if self.anchor is None: self.anchor = t.copy()
        if t.shape[0] >= win.shape[0] or t.shape[1] >= win.shape[1]: return None
        r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, (bx, by) = cv2.minMaxLoc(r)
        if score < self.need: return None
        if self.anchor.shape == t.shape:
            a = cv2.matchTemplate(win[by:by + t.shape[0], bx:bx + t.shape[1]], self.anchor, cv2.TM_CCOEFF_NORMED)[0, 0]
            if a < self.need_anchor: return None                         # looks like something else now: drifted
        b = (bx + x0 + m, by + y0 + m, t.shape[1] - 2 * m, t.shape[0] - 2 * m)
        self._learn(g, b, blend=self.blend)
        return b

    def _blob(self, g, pan, tilt, t_cap):
        w = self._window(g, pan, tilt, t_cap)
        if w is None: return None
        win, x0, y0, u, v = w
        thr = min(np.median(win) - 40, 80)
        if self.level is not None: thr = min(thr, self.level + 18)      # as dark as the target, not a dim poster
        n, lab, st, cen = cv2.connectedComponentsWithStats((win < thr).astype(np.uint8), 8)
        best, bd = None, 1e9
        for i in range(1, n):
            a, bw, bh = st[i, cv2.CC_STAT_AREA], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            if a < 3 or max(bw, bh) > 2.0 * self.size + 4: continue    # specks, or merged with something big
            dd = (cen[i][0] + x0 - u) ** 2 + (cen[i][1] + y0 - v) ** 2
            if dd < bd: bd, best = dd, i
        if best is not None:
            x, y, w, h = st[best, :4]
            box = (int(x + x0), int(y + y0), int(w), int(h))
            self._learn(g, box)
            return box
        return self._match(win, x0, y0)                                # fall back to what the target looks like

    # -- main step
    def _small(self, frame):
        g = gray(frame)
        return g if self.k >= 1.0 else cv2.resize(g, (self.small_cam.w, self.small_cam.h), interpolation=cv2.INTER_AREA)

    def _run_small(self, fn, frame, pan, tilt, t_cap):
        """Run the blob/template finder on the shrunk picture; return the box in full-size pixels."""
        self.cam = self.small_cam
        try:
            b = fn(self._small(frame), pan, tilt, t_cap)
        finally:
            self.cam = self.full_cam
        if b is None: return None, None
        return tuple(v / self.k for v in b), b

    def update(self, frame, t_cap, pan_meas, tilt_meas, t_now):
        self.kp.predict_to(t_cap); self.kt.predict_to(t_cap)
        b_small = None
        if self.kind in ("BLOB", "TEMPLATE"):
            fn = self._blob if self.kind == "BLOB" else self._template
            b, b_small = self._run_small(fn, frame, pan_meas, tilt_meas, t_cap); ok = b is not None
        else:
            ok, b = self.trk.update(bgr(frame))
        if ok:
            cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
            zp, zt = pixel_bearing(self.cam, pan_meas, tilt_meas, cx, cy)
            zp = self.kp.x[0] + ((zp - self.kp.x[0] + 180) % 360 - 180)
            sz = max(b_small[2], b_small[3]) if b_small is not None else max(b[2], b[3])
            if self.hits < WARMUP:
                pass                                               # the picked box is rough: learn the real size first
            elif self.size > 4 and not (self.size / 2.2 <= sz <= self.size * 2.2):
                ok = False                                         # a speck or a big blob: not the same target
            elif math.hypot(zp - self.kp.x[0], zt - self.kt.x[0]) > self.gate:
                ok = False                                         # a jump no drone can make: tracker slipped
        if ok:
            self.misses = 0; self.box = b
            self.kp.correct(zp); self.kt.correct(zt)
            self.size = sz if self.hits == 0 else (0.6 if self.hits < WARMUP else 0.85) * self.size + \
                (0.4 if self.hits < WARMUP else 0.15) * sz
            self.hits += 1
        else:
            self.misses += 1; self.box = None
            self.kp.x[1] *= COAST_DAMP; self.kt.x[1] *= COAST_DAMP   # don't fling the turret along a guess
            if self.misses > self.give_up: return None, False         # lost: stop coasting
        goal = self.predicted(t_now + self.lead)
        locked = ok and math.hypot(goal[0] - pan_meas, goal[1] - tilt_meas) < 2.0
        return goal, locked
