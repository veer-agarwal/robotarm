"""Rough pan-tilt turret mockup from SO-101 parts: base + shoulder (pan) + shoulder-lift horn (tilt),
with the upper arm cut short and a deck carrying the laser and webcam."""
import numpy as np, trimesh
from trimesh.transformations import euler_matrix, translation_matrix, rotation_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

A = "/home/claude/SO-ARM100/Simulation/SO101/assets/"
def T(xyz, rpy): return translation_matrix(xyz) @ euler_matrix(*rpy, "sxyz")
def load(name, xyz, rpy):
    m = trimesh.load(A + name); m.apply_transform(T(xyz, rpy)); return m

base = [load("base_motor_holder_so101_v1.stl", (-0.00636471, 0, -0.0024), (1.5708, 0, 1.5708)),
        load("base_so101_v2.stl", (-0.00636471, 0, -0.0024), (1.5708, 0, 1.5708)),
        load("sts3215_03a_v1.stl", (0.0263353, 0, 0.0437), (0, 0, 0))]
shoulder = [load("sts3215_03a_v1.stl", (-0.0303992, 0.000422, -0.0417), (1.5708, 1.5708, 0)),
            load("motor_holder_so101_base_v1.stl", (-0.0675992, -0.000178, 0.0158499), (1.5708, -1.5708, 0)),
            load("rotation_pitch_so101_v1.stl", (0.0122008, 0, 0.0464), (-1.5708, 0, 0))]

# upper arm stub: keep only the horn end of the stock upper arm (x > -0.03 in link frame)
ua = load("upper_arm_so101_v1.stl", (-0.065085, 0.012, 0.0182), (3.14159, 0, 0))
ua = ua.slice_plane([-0.03, 0, 0], [1, 0, 0], cap=True)

def box(ext, ctr):
    b = trimesh.creation.box(ext); b.apply_translation(ctr); return b

J_pan = T((0.0388353, 0, 0.0624), (3.14159, 0, -3.14159))
J_lift = T((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0))

def pose(q_pan, q_tilt):
    Ts = J_pan @ rotation_matrix(q_pan, [0, 0, 1])
    Tu = Ts @ J_lift @ rotation_matrix(q_tilt, [0, 0, 1])
    return Ts, Tu

# choose the tilt where the stub's -x points horizontally forward
best = min(np.linspace(-1.7, 1.7, 681), key=lambda q: abs((pose(0, q)[1][:3, :3] @ [-1, 0, 0])[2]) + (0 if (pose(0, q)[1][:3, :3] @ [-1, 0, 0])[0] > 0 else 9))
Ts, Tu = pose(0, best)
up_y = np.sign((Tu[:3, :3] @ [0, 1, 0])[2])   # which side of the stub faces up

# deck + payload in the upper-arm frame (forward = -x, up = up_y*y, tilt axis = z)
yb = (ua.bounds[1][1] if up_y > 0 else ua.bounds[0][1])
print("stub bounds", ua.bounds.round(4))
deck = box((0.12, 0.004, 0.06), (-0.085, yb + 0.002 * up_y, ua.centroid[2]))
laser = trimesh.creation.cylinder(radius=0.00525, height=0.019, sections=24)
laser.apply_transform(rotation_matrix(np.pi / 2, [0, 1, 0])); laser.apply_translation((-0.135, yb + 0.0095 * up_y, ua.centroid[2] - 0.016))
cam = box((0.03, 0.03, 0.03), (-0.125, yb + 0.019 * up_y, ua.centroid[2] + 0.014))
lens = trimesh.creation.cylinder(radius=0.007, height=0.006, sections=24)
lens.apply_transform(rotation_matrix(np.pi / 2, [0, 1, 0])); lens.apply_translation((-0.143, yb + 0.019 * up_y, ua.centroid[2] + 0.014))
payload = [deck, laser, cam, lens]

parts = ([(m, "#9aa3ad") for m in base] +
         [(m.copy().apply_transform(Ts), "#c9ced4") for m in shoulder] +
         [(ua.copy().apply_transform(Tu), "#e8b04b")] +
         [(p.copy().apply_transform(Tu), c) for p, c in zip(payload, ["#2E75B6", "#d62728", "#333333", "#111111"])])

for p, c in parts: p.metadata["c"] = c
allv = np.vstack([p.vertices for p, _ in parts])
lo, hi = allv.min(0), allv.max(0); mid = (lo + hi) / 2; r = (hi - lo).max() / 2 * 0.8

fig = plt.figure(figsize=(14, 5.2))
for i, (elev, az, ttl) in enumerate([(22, -55, "3/4 view"), (0, -90, "side (tilt axis into page)"), (90, -90, "top")]):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    for m, c in parts:
        base_rgb = np.array(matplotlib.colors.to_rgb(c))
        L = np.array([0.4, -0.6, 0.7]); L /= np.linalg.norm(L)
        k = 0.45 + 0.55 * np.clip(np.abs(m.face_normals @ L), 0, 1)
        ax.add_collection3d(Poly3DCollection(m.vertices[m.faces], facecolors=np.clip(base_rgb * k[:, None], 0, 1), linewidths=0))
    for k, s in enumerate("xyz"):
        getattr(ax, f"set_{s}lim")(mid[k] - r, mid[k] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, az); ax.set_axis_off(); ax.set_title(ttl)
fig.suptitle("Pan-tilt turret mockup: SO-101 base + shoulder (grey), cut-down upper arm (gold), deck (blue), laser (red), webcam (black)", fontsize=10)
plt.tight_layout(); plt.savefig("/home/claude/turret/turret_mockup.png", dpi=120)

trimesh.util.concatenate([p for p, _ in parts]).export("/home/claude/turret/turret_mockup_assembly.stl")
ext = hi - lo
print("tilt for level deck (deg):", round(np.degrees(best), 1), "up_y", up_y)
print("overall size mm:", (ext * 1000).round(0))
