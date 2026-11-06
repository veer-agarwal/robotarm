"""Turret HUD, design C ("tapes"): pan/tilt angle tapes on the edges, a diamond on the target, a lead mark where
it's aiming, and a SEARCH / TRACK / LOCK / FIRE ladder with a hold bar along the bottom.

Also the fire rule (FireControl): FIRE only when armed, locked for HOLD_MS straight, the aim well inside the
target's outline, and the prediction steady (seen in 6 frames straight). Lose any of them and it drops back.
"Inside the outline" scales with distance: a drone 3 m away is ~2 deg across, so LOCK = aim within 1.2x its
half-width, FIRE = within 0.6x of it. With no size known it uses LOCK_DEG / FIRE_DEG.

Everything is drawn on an upscaled copy of the frame, so the lines stay crisp on the 640x400 Arducam picture and
recordings stay clean.
"""
import math
import time
import cv2
import numpy as np

# colours (BGR) — cyan = locked, orange = fire; they differ in brightness as well as hue
WHITE = (242, 244, 245)
DIM = (150, 152, 154)
LOCK = (255, 230, 94)        # #5ee6ff
FIRE = (58, 162, 255)        # #ffa23a
INK = (19, 18, 17)
FONT = cv2.FONT_HERSHEY_SIMPLEX

LOCK_DEG = 1.0               # aim error to count as locked (if the target's size isn't known)
FIRE_DEG = 0.5               # aim error allowed when firing (same)
LOCK_K, FIRE_K = 1.2, 0.6    # ... else these times the target's half-width in degrees
HOLD_MS = 150                # locked this long before it may fire
STATES = ("SEARCH", "TRACK", "LOCK", "FIRE")


class FireControl:
    """Turns tracking numbers into a state. update() each frame; .state, .hold_ms, .reason."""

    def __init__(self, lock_deg=LOCK_DEG, fire_deg=FIRE_DEG, hold_ms=HOLD_MS):
        self.lock_deg, self.fire_deg, self.hold_ms_need = lock_deg, fire_deg, hold_ms
        self.lock_since = None; self.state = "SEARCH"; self.hold_ms = 0.0; self.reason = "NO TARGET"
        self.ready = False

    def update(self, tracking, err_deg, steady, armed, half_deg=None, now=None):
        now = time.perf_counter() if now is None else now
        if half_deg:
            self.lock_deg, self.fire_deg = max(0.3, LOCK_K * half_deg), max(0.15, FIRE_K * half_deg)
        if not tracking:
            self.lock_since = None; self.hold_ms = 0; self.ready = False
            self.state, self.reason = "SEARCH", "NO TARGET - PRESS C"
            return self.state
        if err_deg is None or err_deg > self.lock_deg:
            self.lock_since = None; self.hold_ms = 0; self.ready = False
            self.state, self.reason = "TRACK", "CENTRE THE TARGET"
            return self.state
        if self.lock_since is None: self.lock_since = now
        self.hold_ms = 1000 * (now - self.lock_since)
        self.ready = self.hold_ms >= self.hold_ms_need and err_deg <= self.fire_deg and steady
        if self.ready and armed:
            self.state, self.reason = "FIRE", f"FIRING  ERR {err_deg:.1f} DEG"
        elif self.ready:
            self.state, self.reason = "LOCK", "WOULD FIRE - SAFE, NOT ARMED (L)"
        else:
            self.state = "LOCK"
            why = [f"HOLD {min(self.hold_ms, self.hold_ms_need):.0f} / {self.hold_ms_need} MS"]
            if err_deg > self.fire_deg: why.append(f"ERR {err_deg:.2f} > {self.fire_deg:.2f}")
            if not steady: why.append("PREDICTION SETTLING")
            self.reason = "  ".join(why)
        return self.state


def _text(img, s, org, scale, col, thick=1, anchor="l"):
    (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    if anchor == "c": x -= w // 2
    elif anchor == "r": x -= w
    if col != INK:                                       # dark edge so text reads on a bright wall
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1)):
            cv2.putText(img, s, (int(x) + dx, int(y) + dy), FONT, scale, INK, thick, cv2.LINE_AA)
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, col, thick, cv2.LINE_AA)
    return w, h


def _shade(img, x0, y0, x1, y1, alpha=0.55):
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(img.shape[1], int(x1)), min(img.shape[0], int(y1))
    if x1 <= x0 or y1 <= y0: return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha)).astype(img.dtype)


def draw(frame, st, out_w=1280):
    """Return the HUD image. st = dict:
        state, reason, hold_ms, armed, pan, tilt (deg), fps, tracker, mode,
        box (x,y,w,h in frame px or None), lead (x,y frame px or None), err (deg or None), rate (deg/s or None),
        recording (bool), exposure"""
    H, W = frame.shape[:2]
    k = out_w / W
    img = cv2.resize(frame, (out_w, int(round(H * k))), interpolation=cv2.INTER_LINEAR)
    if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    OH, OW = img.shape[:2]
    u = OW / 1280.0                                      # design units (the mock-up was 1280 wide)
    state = st.get("state", "SEARCH")
    armed = st.get("armed", False)
    col = {"SEARCH": WHITE, "TRACK": WHITE, "LOCK": LOCK, "FIRE": FIRE}[state]
    ts = 0.5 * u                                        # small text scale

    # -- pan tape (top centre): +-30 deg, tick every 5, label every 10
    tw, th = 640 * u, 64 * u
    tx0 = OW / 2 - tw / 2
    _shade(img, tx0, 0, tx0 + tw, th)
    pan = st.get("pan", 0.0)
    for a in range(int(math.floor((pan - 30) / 5) * 5), int(pan + 31), 5):
        x = OW / 2 + (a - pan) * (tw / 60)
        if x < tx0 + 2 or x > tx0 + tw - 2: continue
        major = a % 10 == 0
        cv2.line(img, (int(x), 0), (int(x), int((18 if major else 10) * u)), DIM if not major else WHITE, 1, cv2.LINE_AA)
        if major: _text(img, str(a), (x, 36 * u), ts, WHITE, anchor="c")
    bx, by = OW / 2 - 36 * u, 42 * u
    cv2.rectangle(img, (int(bx), int(by)), (int(bx + 72 * u), int(by + 22 * u)), INK, -1)
    cv2.rectangle(img, (int(bx), int(by)), (int(bx + 72 * u), int(by + 22 * u)), col, 1)
    _text(img, f"{pan:+.1f}", (OW / 2, by + 16 * u), ts, col, anchor="c")

    # -- tilt tape (right edge): +-20 deg over 400 design px, centred vertically
    vh = min(400 * u, OH - 220 * u)
    vx0, vy0 = OW - 64 * u, OH / 2 - vh / 2
    _shade(img, vx0, vy0, OW, vy0 + vh)
    tilt = st.get("tilt", 0.0)
    for a in range(int(math.floor((tilt - 20) / 5) * 5), int(tilt + 21), 5):
        y = OH / 2 - (a - tilt) * (vh / 40)
        if y < vy0 + 2 or y > vy0 + vh - 2: continue
        major = a % 10 == 0
        cv2.line(img, (int(OW - (18 if major else 10) * u), int(y)), (OW, int(y)), WHITE if major else DIM, 1, cv2.LINE_AA)
        if major: _text(img, str(a), (vx0 + 4 * u, y + 5 * u), ts, WHITE)
    bx, by = OW - 130 * u, OH / 2 - 11 * u
    cv2.rectangle(img, (int(bx), int(by)), (int(bx + 64 * u), int(by + 22 * u)), INK, -1)
    cv2.rectangle(img, (int(bx), int(by)), (int(bx + 64 * u), int(by + 22 * u)), col, 1)
    _text(img, f"{tilt:+.1f}", (bx + 32 * u, by + 16 * u), ts, col, anchor="c")

    # -- crosshair: two bars with a gap
    cx, cy = OW // 2, OH // 2
    cv2.line(img, (int(cx - 30 * u), cy), (int(cx - 8 * u), cy), WHITE, 2, cv2.LINE_AA)
    cv2.line(img, (int(cx + 8 * u), cy), (int(cx + 30 * u), cy), WHITE, 2, cv2.LINE_AA)

    # -- target diamond, lead cross, numbers
    box = st.get("box")
    if box is not None and state != "SEARCH":
        x, y, w, h = box
        mx, my = (x + w / 2) * k, (y + h / 2) * k
        r = max(28 * u, 0.75 * max(w, h) * k + 10 * u)
        pts = np.array([[mx, my - r], [mx + r, my], [mx, my + r], [mx - r, my]], np.int32)
        cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
        info = []
        if st.get("err") is not None: info.append(f"ERR {st['err']:.1f}")
        if st.get("rate") is not None: info.append(f"{st['rate']:.0f} DEG/S")
        for i, s in enumerate(info):
            _text(img, s, (mx + r + 10 * u, my - r / 2 + i * 18 * u), ts, col)
    lead = st.get("lead")
    if lead is not None and state != "SEARCH":
        lx, ly = lead[0] * k, lead[1] * k
        cv2.line(img, (int(lx - 6 * u), int(ly)), (int(lx + 6 * u), int(ly)), col, 2, cv2.LINE_AA)
        cv2.line(img, (int(lx), int(ly - 6 * u)), (int(lx), int(ly + 6 * u)), col, 2, cv2.LINE_AA)

    # -- top-left status
    _text(img, f"{st.get('mode', '').upper()}  {st.get('tracker', '') or '-'}  {st.get('fps', 0):.0f} FPS", (32 * u, 92 * u), ts, WHITE)
    _text(img, "ARMED" if armed else "SAFE", (32 * u, 114 * u), ts, FIRE if armed else DIM)
    if st.get("recording"):
        cv2.circle(img, (int(32 * u + 6 * u), int(132 * u)), int(5 * u), (60, 60, 255), -1, cv2.LINE_AA)
        _text(img, "REC", (32 * u + 16 * u, 137 * u), ts, (60, 60, 255))

    # -- bottom ladder + hold bar + reason
    seg_w, seg_h, gap = 120 * u, 40 * u, 2 * u
    lw = 4 * seg_w + 3 * gap
    lx0, ly0 = OW / 2 - lw / 2, OH - 28 * u - 18 * u - 14 * u - seg_h
    idx = STATES.index(state)
    for i, name in enumerate(STATES):
        x0 = lx0 + i * (seg_w + gap)
        here = i == idx
        c = (FIRE if armed else LOCK) if name == "FIRE" else (LOCK if name == "LOCK" else WHITE)
        if here:
            cv2.rectangle(img, (int(x0), int(ly0)), (int(x0 + seg_w), int(ly0 + seg_h)), c, -1)
        else:
            _shade(img, x0, ly0, x0 + seg_w, ly0 + seg_h, 0.55 if i > idx else 0.25)
        label = "READY" if name == "FIRE" and not armed else name
        _text(img, label, (x0 + seg_w / 2, ly0 + seg_h / 2 + 6 * u), 0.5 * u, INK if here else (WHITE if i < idx else DIM),
              2 if here else 1, anchor="c")
    hy = ly0 + seg_h + 10 * u
    cv2.rectangle(img, (int(lx0), int(hy)), (int(lx0 + lw), int(hy + 4 * u)), (70, 70, 70), -1)
    frac = min(1.0, st.get("hold_ms", 0) / HOLD_MS) if state in ("LOCK", "FIRE") else 0.0
    if frac > 0:
        cv2.rectangle(img, (int(lx0), int(hy)), (int(lx0 + lw * frac), int(hy + 4 * u)), col, -1)
    _text(img, st.get("reason", ""), (OW / 2, hy + 22 * u), ts, WHITE, anchor="c")
    return img
