import sys
sys.argv.append("--no-export")
import numpy as np, trimesh, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from trimesh.transformations import rotation_matrix as Rot
import perses_turret as T

COL = {"base": "#8a939e", "pan_floor_cap": "#5b6470", "turntable": "#b9c1ca", "tower_servo": "#2E75B6",
       "tilt_floor_cap": "#5b6470", "tower_bearing": "#2E75B6", "cradle": "#e8a33d", "shaft_washer": "#5b6470",
       "servo_pan": "#262626", "servo_tilt": "#262626", "perses": "#d9534f", "laser_plate": "#e8a33d", "laser": "#d62728", "webcam": "#262626", "bearing_tilt": "#d4af37"}
L = np.array([0.35, -0.55, 0.75]); L /= np.linalg.norm(L)

def scene(pan=0, tilt=0, perses=True):
    out = []
    Mt = Rot(np.radians(tilt), (0, 1, 0), point=(0, 0, T.ZT)); Mp = Rot(np.radians(pan), (0, 0, 1))
    for n, m in {**T.parts, **T.hardware}.items():
        if n in T.PAYLOAD_HW and not perses: continue
        mm = m.copy()
        if n in T.TILTING: mm.apply_transform(Mt)
        if n not in T.STATIC: mm.apply_transform(Mp)
        c = COL.get(n, "#d4af37" if "bearing" in n else "#5b6470")
        out.append((mm, c))
    return out

def draw(ax, sc, elev, azim, title, alpha_perses=None):
    tris, cols = [], []
    for m, c in sc:
        k = 0.42 + 0.58 * np.clip(np.abs(m.face_normals @ L), 0, 1)
        rgb = np.clip(np.array(matplotlib.colors.to_rgb(c)) * k[:, None], 0, 1)
        tris.append(m.vertices[m.faces]); cols.append(rgb)
    tris = np.concatenate(tris); cols = np.concatenate(cols)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, linewidths=0))
    v = tris.reshape(-1, 3); lo, hi = v.min(0), v.max(0); mid = (lo + hi) / 2; r = (hi - lo).max() / 2 * 0.82
    for k, s in enumerate("xyz"): getattr(ax, f"set_{s}lim")(mid[k] - r, mid[k] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, azim); ax.set_axis_off(); ax.set_title(title, fontsize=10)

fig = plt.figure(figsize=(16, 6))
draw(fig.add_subplot(1, 3, 1, projection="3d"), scene(0, 0), 18, -42, f"Assembly, payload: {T.PAYLOAD}")
draw(fig.add_subplot(1, 3, 2, projection="3d"), scene(0, T.TILT_LIMIT), 0, -90, f"Side, tilted up {T.TILT_LIMIT:.0f} deg (limit)")
draw(fig.add_subplot(1, 3, 3, projection="3d"), scene(0, 0), 0, 0, "Front")
fig.suptitle("Perses pan-tilt turret: base + pan servo (grey/black), turntable on 3x 6802 rollers, towers (blue), "
             "tilt servo + 6802 idler, cradle (orange)", fontsize=10)
plt.tight_layout(); plt.savefig(f"render_assembly_{T.PAYLOAD}.png", dpi=115); plt.close()

fig = plt.figure(figsize=(16, 6))
draw(fig.add_subplot(1, 3, 1, projection="3d"), scene(0, 0, False), 22, -42, "Mechanism (Perses hidden)")
draw(fig.add_subplot(1, 3, 2, projection="3d"), scene(0, 0, False), 22, 138, "Back: tilt servo sleeve + floor cap")
draw(fig.add_subplot(1, 3, 3, projection="3d"), scene(45, -20, False), 18, -60, "Panned 45, tilted down 20")
plt.tight_layout(); plt.savefig("render_mechanism.png", dpi=115); plt.close()

# printable parts sheet
import glob, os
files = sorted(glob.glob(os.path.join(T.OUT, "*.stl")))
fig = plt.figure(figsize=(16, 12))
for i, f in enumerate(files):
    m = trimesh.load(f); ax = fig.add_subplot(3, 4, i + 1, projection="3d")
    draw(ax, [(m, "#e8a33d")], 28, -55, f"{os.path.basename(f)}\n{np.round(m.extents).astype(int)} mm")
plt.tight_layout(); plt.savefig("render_parts.png", dpi=105); plt.close()
print("rendered")
