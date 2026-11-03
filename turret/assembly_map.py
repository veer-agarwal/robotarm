"""Exploded assembly pictures for the small turret parts: rollers, tilt bearing side, tilt horn side."""
import os
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import trimesh
import perses_turret as P

OUT = os.environ.get("ASSY_OUT", os.path.join(P.HERE, "out", "assembly"))
os.makedirs(OUT, exist_ok=True)

C_PRINT = np.array([0.80, 0.84, 0.90])
C_PART = np.array([0.98, 0.72, 0.40])     # the small part being installed
C_BRG = np.array([0.72, 0.74, 0.76])
C_SCREW = np.array([0.30, 0.31, 0.34])
C_SERVO = np.array([0.20, 0.22, 0.26])


def screw_mesh(head, ax, L):
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    h = trimesh.creation.cylinder(radius=2.75, segment=np.array([head - ax * 3.0, head]), sections=20)
    s = trimesh.creation.cylinder(radius=1.5, segment=np.array([head, head + ax * L]), sections=16)
    return trimesh.util.concatenate([h, s])


def crop(mesh, **lim):
    """Cut the mesh with capped planes: crop(m, ymin=25, xmax=40, ...)."""
    m = mesh.copy()
    axes = {"x": 0, "y": 1, "z": 2}
    for k, v in lim.items():
        n = np.zeros(3); n[axes[k[0]]] = 1.0 if k.endswith("min") else -1.0
        o = np.zeros(3); o[axes[k[0]]] = v
        m = m.slice_plane(o, n, cap=True)
    return m


def render_multi(ax, items, d, up_hint=(0, 0, 1)):
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    up = np.asarray(up_hint, float); up = up - (up @ d) * d; up /= np.linalg.norm(up)
    right = np.cross(d, up)
    polys, depths, fcs, ecs, lws = [], [], [], [], []
    allxy = []
    for mesh, color in items:
        tri, n = mesh.triangles, mesh.face_normals
        vis = n @ d < 0
        t, nn = tri[vis], n[vis]
        dep = t.mean(axis=1) @ d
        sh = 0.42 + 0.58 * np.abs(nn @ d)
        for tt, s, dd in zip(t, sh, dep):
            c = tuple(np.clip(color * s, 0, 1))
            polys.append(np.stack([tt @ right, tt @ up], -1)); depths.append(dd); fcs.append(c); ecs.append(c); lws.append(0.3)
        allxy.append(np.stack([t.reshape(-1, 3) @ right, t.reshape(-1, 3) @ up], -1))
        if len(mesh.face_adjacency):
            sharp = mesh.face_adjacency_angles > np.radians(28)
            fa = mesh.face_adjacency[sharp]
            v = vis[fa[:, 0]] | vis[fa[:, 1]]
            for e in mesh.vertices[mesh.face_adjacency_edges[sharp][v]]:
                polys.append(np.stack([e @ right, e @ up], -1)); depths.append(e.mean(0) @ d - 0.4)
                fcs.append("none"); ecs.append("#2b2f38"); lws.append(0.8)
    order = np.argsort(-np.array(depths))
    ax.add_collection(PolyCollection([polys[i] for i in order], facecolors=[fcs[i] for i in order],
                                     edgecolors=[ecs[i] for i in order], linewidths=[lws[i] for i in order], closed=False))
    xy = np.concatenate(allxy)
    pad = 0.08 * (xy.max(0) - xy.min(0)).max()
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal"); ax.axis("off")
    return lambda p: (float(np.dot(p, right)), float(np.dot(p, up)))


def tag(ax, proj, p, text, dx, dy, color="#111"):
    x, y = proj(p)
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), fontsize=10.5, weight="bold", color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", lw=0.8),
                arrowprops=dict(arrowstyle="-", color="#666", lw=1), zorder=20)


def path(ax, proj, a, b):
    (x0, y0), (x1, y1) = proj(a), proj(b)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#d0452b", ls="--"), zorder=15)


def page(title, steps, draw, fn, size=(13, 9.5)):
    fig = plt.figure(figsize=size, dpi=140)
    fig.suptitle(title, fontsize=19, weight="bold", y=0.975)
    ax = fig.add_axes([0.02, 0.27, 0.96, 0.58])
    draw(ax)
    txt = "\n".join(textwrap.fill(f"{i}. {s}", 150, subsequent_indent="    ") for i, s in enumerate(steps, 1))
    fig.text(0.05, 0.235, txt, fontsize=11.5, va="top", linespacing=1.55)
    fig.savefig(os.path.join(OUT, fn), facecolor="white"); plt.close(fig); print("wrote", fn)


SCREWS = P.screws()


def screws_named(prefix):
    return [(np.asarray(h, float), np.asarray(a, float), L) for name, L, h, a, _ in SCREWS if name.startswith(prefix)]


# ---------------------------------------------------------------- 1. roller (post at 270 deg, repeat x3)
def rollers():
    ang = 270
    M = P.roller_M(ang)
    out = M[:3, :3] @ [0, 0, 1.0]                        # radial outward
    tt = crop(P.turntable(), xmin=-45, xmax=45, ymax=-35)
    axle = P.roller_axle(); axle.apply_transform(M)
    brg = P.bearing(); brg.apply_transform(M @ P.Tr((0, 0, 2.0 + P.BRG_W / 2 + 0.2)))
    head, sax, L = [s for s in screws_named("roller axle") if abs(np.arctan2(s[0][1], s[0][0]) - np.radians(-90)) < 0.3][0]
    E = {"axle": 16, "brg": 34, "screw": 54}
    axle.apply_translation(out * E["axle"]); brg.apply_translation(out * E["brg"])
    sc = screw_mesh(head + out * E["screw"], sax, L)
    post = np.array((0, -P.ROLL_R, P.ROLL_Z))

    def draw(ax):
        proj = render_multi(ax, [(tt, C_PRINT), (axle, C_PART), (brg, C_BRG), (sc, C_SCREW)], d=(0.55, 0.62, 0.56))
        path(ax, proj, post + out * 78, post + out * 2)
        tag(ax, proj, post + out * (E["screw"] + 6), "M3x12", 0, -28)
        tag(ax, proj, post + out * (E["brg"] + 2), "6802 bearing", 30, 34)
        tag(ax, proj, post + out * (E["axle"] + 3), "roller_axle\n(flange toward the post)", -40, -30, color="#b5651d")
        tag(ax, proj, post + np.array((0, 0, -4)), "post under the turntable\n(insert 17/18/19)", -55, 30)

    page("Rollers: 3 the same (one shown)",
         ["Press a 6802 onto the roller_axle's stub (the narrow side) until it touches the flange. Same fit as your test.",
          "Hold the axle FLANGE-side against the post's outer face, centered on the insert.",
          "M3x12 through the middle of the axle into the insert. Snug, don't crank it (the axle is plastic).",
          "Spin the bearing with a finger: the outer ring should turn freely. Repeat for the other 2 posts."],
         draw, "1_rollers_assembly.png")


# ---------------------------------------------------------------- 2. tilt axis, bearing side
def tilt_bearing_side():
    y = np.array((0, 1.0, 0))
    plate = crop(P.laser_plate(), ymin=25, xmin=-40, xmax=40)
    tower = crop(P.tower_bearing(), zmin=P.ZT - 38)
    stub = P.stub_shaft()
    brg = P.hardware["bearing_tilt"].copy()
    wash = P.shaft_washer()
    fl = [screw_mesh(h, a, L) for h, a, L in screws_named("stub flange")]
    wh, wa, wl = screws_named("washer")[0]
    E = {"fl": -46, "stub": -26, "tower": 22, "brg": 44, "wash": 60, "ws": 74}
    stub.apply_translation(y * E["stub"]); tower.apply_translation(y * E["tower"])
    brg.apply_translation(y * E["brg"]); wash.apply_translation(y * E["wash"])
    for m in fl: m.apply_translation(y * E["fl"])
    ws = screw_mesh(wh + y * E["ws"], wa, wl)
    ax0 = np.array((0, 0, P.ZT))

    def draw(ax):
        proj = render_multi(ax, [(plate, C_PRINT), (tower, C_PRINT), (stub, C_PART), (brg, C_BRG), (wash, C_PART), (ws, C_SCREW)] +
                            [(m, C_SCREW) for m in fl], d=(-0.78, 0.18, -0.6))
        path(ax, proj, ax0 + y * 15, ax0 + y * 40)
        path(ax, proj, ax0 + y * 100, ax0 + y * 60)
        tag(ax, proj, ax0 + y * (36 + E["fl"]) + np.array((0, 0, 14)), "3x M3x8", -45, 8)
        tag(ax, proj, ax0 + y * (44 + E["stub"]), "stub_shaft", 0, -48, color="#b5651d")
        tag(ax, proj, ax0 + y * 42 + np.array((0, 0, 18)), "laser_plate ear\n(inserts 2-4 on this inner face)", -75, 30)
        tag(ax, proj, ax0 + y * (50 + E["tower"]) + np.array((0, 0, 25)), "tower_bearing", -20, 40)
        tag(ax, proj, ax0 + y * (50 + E["brg"]), "6802", 0, -42)
        tag(ax, proj, ax0 + y * (54.7 + E["wash"]), "shaft_washer", 45, 60, color="#b5651d")
        tag(ax, proj, ax0 + y * (55.7 + E["ws"]), "M3x6", 20, -34)

    page("Tilt axis, bearing side: stub_shaft + bearing + washer",
         ["Bolt the stub_shaft to the laser plate's ear: flange on the INSIDE face, shaft sticking out through the ear's big hole, 3x M3x8 into inserts 2-4.",
          "Press a 6802 into tower_bearing's seat from the OUTSIDE face (the seat opens outward). Push on the outer ring only.",
          "Later, when the towers are on the turntable: slide the stub shaft into the bearing from the inside (this happens with step 2 of the horn-side picture).",
          "From the outside: shaft_washer on the end of the shaft, M3x6 into the stub's tip insert. Snug only: the plate must still tilt freely."],
         draw, "2_tilt_bearing_side.png")


# ---------------------------------------------------------------- 3. tilt axis, servo horn side
def tilt_horn_side():
    y = np.array((0, 1.0, 0))
    plate = crop(P.laser_plate(), ymax=-25, xmin=-40, xmax=40)
    tower = crop(P.tower_servo(), zmin=P.ZT - 40)
    servo = crop(P.hardware["servo_tilt"], zmin=P.ZT - 40)
    spacer = P.horn_spacer()
    hs = [screw_mesh(h, a, L) for h, a, L in screws_named("cradle ear + spacer")]
    E = {"sc": 40, "spacer": -16, "tower": -34}
    spacer.apply_translation(y * E["spacer"]); tower.apply_translation(y * E["tower"]); servo.apply_translation(y * E["tower"])
    for m in hs: m.apply_translation(y * E["sc"])
    ax0 = np.array((0, 0, P.ZT))

    def draw(ax):
        proj = render_multi(ax, [(plate, C_PRINT), (tower, C_PRINT), (servo, C_SERVO), (spacer, C_PART)] + [(m, C_SCREW) for m in hs],
                            d=(-0.78, -0.18, -0.6))
        path(ax, proj, ax0 - y * 0 + y * 10, ax0 - y * 60)
        tag(ax, proj, ax0 + y * (-36 + E["sc"]) + np.array((0, 0, 14)), "4x M3x8", 0, 42)
        tag(ax, proj, ax0 - y * 42 + np.array((0, 0, 18)), "laser_plate ear\n(servo side)", 10, 48)
        tag(ax, proj, ax0 + y * (-46.5 + E["spacer"]), "horn_spacer", 0, -46, color="#b5651d")
        tag(ax, proj, ax0 + y * (-50 + E["tower"]) + np.array((0, 0, 22)), "tilt servo horn\n(in tower_servo)", -95, 15)

    page("Tilt axis, servo side: horn_spacer between the servo horn and the plate",
         ["Center the tilt servo first (turret_setup center: 2048), with tower_servo bolted to the turntable and the servo in its sleeve.",
          "Hold the laser plate level (laser pointing forward). Stub shaft side goes into the bearing tower, this side sits over the servo horn.",
          "horn_spacer between the horn and the ear, its 4 holes lined up with the horn's 4 threaded holes.",
          "4x M3x8 from the INSIDE of the ear, through the ear and the spacer into the horn. Then finish the washer + M3x6 on the bearing side."],
         draw, "3_tilt_servo_side.png")


if __name__ == "__main__":
    rollers(); tilt_bearing_side(); tilt_horn_side()
