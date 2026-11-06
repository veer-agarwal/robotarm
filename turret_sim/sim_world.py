"""Turret simulator: room, Tello flight paths, the Arducam OV9281 camera, and STS3215 servos.

Everything is in degrees and seconds. Pan = rotation about the vertical axis (+ = left, seen from behind the turret),
tilt = elevation (+ = up). The camera rides on the tilt axis, looking along the turret's aim.
"""
import math
import numpy as np
import cv2

# ---------------------------------------------------------------- camera (Arducam OV9281, 130 deg diagonal M12 lens)
class Camera:
    """Rectilinear model of the wide lens, binned to 640x400 so tracking runs fast. Global shutter, mono."""

    def __init__(self, w=640, h=400, diag_fov=130.0, fps=120, latency=0.025, exposure=0.002, noise=4.0):
        self.w, self.h, self.fps = w, h, fps
        self.f = math.hypot(w, h) / 2 / math.tan(math.radians(diag_fov / 2))   # focal length in pixels
        self.cx, self.cy = w / 2, h / 2
        self.hfov = math.degrees(2 * math.atan(w / 2 / self.f))
        self.vfov = math.degrees(2 * math.atan(h / 2 / self.f))
        self.latency = latency          # capture -> frame in the tracker's hands (exposure, readout, USB, decode)
        self.exposure, self.noise = exposure, noise
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        self.rays = np.stack([np.ones_like(u) * self.f, -(u - self.cx), -(v - self.cy)], -1)   # x fwd, y left, z up
        self.rays /= np.linalg.norm(self.rays, axis=-1, keepdims=True)

    def pixel_to_angles(self, px, py):
        """Pixel -> (degrees left of the aim, degrees above the aim), exact for this lens model."""
        return math.degrees(math.atan2(self.cx - px, self.f)), math.degrees(math.atan2(self.cy - py, math.hypot(self.f, px - self.cx)))


def rot(pan, tilt):
    """World-from-camera rotation for turret angles in degrees."""
    p, t = math.radians(pan), math.radians(tilt)
    Rz = np.array([[math.cos(p), -math.sin(p), 0], [math.sin(p), math.cos(p), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(t), 0, -math.sin(t)], [0, 1, 0], [math.sin(t), 0, math.cos(t)]])
    return Rz @ Ry


def bearing(p):
    """(pan, tilt) in degrees that point at world point p from the turret at the origin."""
    return math.degrees(math.atan2(p[1], p[0])), math.degrees(math.atan2(p[2], math.hypot(p[0], p[1])))


# ---------------------------------------------------------------- room texture (what the camera sees behind the drone)
def make_room(seed=0, W=2048, H=1024):
    """Equirectangular grey panorama: walls with posters and shelves, bright ceiling lights and a window, darker floor."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 150, np.float32)
    img[H // 2 + 60:] = 95                                   # floor
    img[:H // 4] = 185                                       # ceiling
    img += cv2.GaussianBlur(rng.normal(0, 18, (H, W)).astype(np.float32), (0, 0), 6)
    for _ in range(120):                                     # posters, shelves, furniture blocks
        x, y = rng.integers(0, W), rng.integers(H // 4, H // 2 + 80)
        w, h = rng.integers(20, 220), rng.integers(15, 140)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), float(rng.integers(40, 230)), -1)
    for _ in range(10):                                      # ceiling lights
        cv2.ellipse(img, (int(rng.integers(0, W)), int(rng.integers(20, H // 4))), (40, 12), 0, 0, 360, 250, -1)
    cv2.rectangle(img, (int(W * 0.62), int(H * 0.3)), (int(W * 0.72), int(H * 0.5)), 240, -1)       # window
    return np.clip(cv2.GaussianBlur(img, (0, 0), 1.2), 0, 255).astype(np.float32)


# ---------------------------------------------------------------- Tello flight paths (turret at origin, room ~5 x 4 m)
def scenarios():
    """name -> function(t) giving the Tello position in metres. Indoor speeds."""
    def hover(t):
        return np.array([2.5 + 0.15 * math.sin(0.7 * t), 0.3 * math.sin(0.5 * t), 0.6 + 0.1 * math.sin(1.1 * t)])

    def pingpong(t, half, speed):
        """Back and forth between -half and +half at speed, easing through each turnaround (no teleporting)."""
        period = 2 * math.pi * half / speed                       # top speed (mid-pass) = speed
        ph = (t % period) / period
        x = 4 * ph - 1 if ph < 0.5 else 3 - 4 * ph               # triangle wave, -1..1
        return half * math.sin(math.pi / 2 * x)

    def crossing(t):                                         # passes across the room, 3 m/s, 2.5 m away
        return np.array([2.5, -pingpong(t + 1.57, 3.0, 3.0), 0.5])

    def zigzag(t):                                           # side-to-side dodge, 1.2 m swings every 1.5 s, 3 m away
        return np.array([3.0, 1.2 * math.sin(2 * math.pi * t / 1.5), 0.5 + 0.2 * math.sin(2 * math.pi * t / 3)])

    def close_pass(t):                                       # fast pass 1.2 m away at 4 m/s: the hardest case
        return np.array([1.2, -pingpong(t + 1.18, 3.0, 4.0), 0.3])

    def orbit(t):                                            # sweeps a wide arc around the turret at 2 m, bobbing
        a = 1.2 * math.sin(0.6 * t)                          # +-69 deg, up to about 1.4 m/s
        return np.array([2.0 * math.cos(a), 2.0 * math.sin(a), 0.5 + 0.3 * math.sin(1.3 * t)])

    def jink(t):                                             # hovers, then darts 1 m sideways in 0.4 s, repeatedly
        k, f = divmod(t, 1.6)
        side = 1 if int(k) % 2 == 0 else -1
        s = min(1.0, f / 0.4); s = s * s * (3 - 2 * s)
        return np.array([2.8, side * (-0.5 + s), 0.6])

    return {"hover": hover, "crossing": crossing, "zigzag": zigzag, "jink": jink, "orbit": orbit, "close_pass": close_pass}


# ---------------------------------------------------------------- STS3215 servo (position mode with goal speed + ACC)
class Servo:
    """Trapezoidal move toward the goal, limited by the speed and acceleration registers, with a little backlash."""

    def __init__(self, vmax=220.0, amax=527.0, backlash=0.15, limits=(-170, 170)):
        self.pos = self.vel = self.goal = 0.0
        self.motor = 0.0                                    # motor side, before the gear slop
        self.vmax, self.amax, self.backlash, self.limits = vmax, amax, backlash, limits

    def step(self, dt):
        e = self.goal - self.motor
        v_stop = math.copysign(math.sqrt(2 * self.amax * abs(e)), e) if e else 0.0
        v_target = max(-self.vmax, min(self.vmax, v_stop))
        dv = max(-self.amax * dt, min(self.amax * dt, v_target - self.vel))
        self.vel += dv
        self.motor += self.vel * dt
        if abs(self.goal - self.motor) < 0.05 and abs(self.vel) < 5:
            self.motor, self.vel = self.goal, 0.0
        half = self.backlash / 2                             # output lags the motor by up to half the slop either way
        if self.motor - self.pos > half: self.pos = self.motor - half
        elif self.pos - self.motor > half: self.pos = self.motor + half
        self.pos = max(self.limits[0], min(self.limits[1], self.pos))


# ---------------------------------------------------------------- rendering
DRONE_SPAN = 0.20       # Tello with prop guards, metres


def render(cam, room, pan, tilt, drone_pts, rng):
    """One camera frame. drone_pts = drone positions sampled across the exposure (for motion blur)."""
    R = rot(pan, tilt)
    d = cam.rays @ R.T
    lon = np.arctan2(d[..., 1], d[..., 0]); lat = np.arcsin(np.clip(d[..., 2], -1, 1))
    H, W = room.shape
    mx = ((np.pi - lon) / (2 * np.pi) * W).astype(np.float32) % W
    my = ((np.pi / 2 - lat) / np.pi * H).astype(np.float32)
    img = cv2.remap(room, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    layer = np.zeros_like(img); cover = np.zeros_like(img)
    Rt = R.T
    for p in drone_pts:
        c = Rt @ p
        if c[0] < 0.2: continue
        u = cam.cx - cam.f * c[1] / c[0]; v = cam.cy - cam.f * c[2] / c[0]
        r = max(1.0, cam.f * DRONE_SPAN / 2 / c[0])
        one = np.zeros_like(img)
        ui, vi, ri = int(round(u * 4)), int(round(v * 4)), int(round(r * 4))
        cv2.ellipse(one, (ui, vi), (ri, int(ri * 0.35)), 0, 0, 360, 1.0, -1, cv2.LINE_AA, 2)      # body + guards
        cv2.ellipse(one, (ui, vi - int(ri * 0.15)), (int(ri * 0.5), int(ri * 0.3)), 0, 0, 360, 1.0, -1, cv2.LINE_AA, 2)
        cover += one
    if drone_pts:
        cover = np.clip(cover / len(drone_pts), 0, 1)
        img = img * (1 - cover) + 35 * cover               # dark grey drone
    img += rng.normal(0, cam.noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def project(cam, pan, tilt, p):
    """Pixel position and radius of the drone, or None if behind the camera."""
    c = rot(pan, tilt).T @ p
    if c[0] < 0.2: return None
    return cam.cx - cam.f * c[1] / c[0], cam.cy - cam.f * c[2] / c[0], cam.f * DRONE_SPAN / 2 / c[0]
