"""Heat-set insert map: one easy-to-read picture per part, numbered circles where each insert goes."""
import os
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import perses_turret as P

OUT = os.environ.get("INSERT_OUT", os.path.join(P.HERE, "out", "inserts"))
os.makedirs(OUT, exist_ok=True)
COL = ["#e4572e", "#2e86de", "#17a34a", "#9b59b6"]
BODY = np.array([0.80, 0.83, 0.88])


def pt(m, p):
    return (m @ np.array([*p, 1.0]))[:3]


def vec(m, v):
    return m[:3, :3] @ np.asarray(v, float)


def render(ax, mesh, d, up, keep=None):
    """Shaded view of `mesh` from a camera looking along d (painter's algorithm)."""
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    up = np.asarray(up, float); right = np.cross(d, up)
    tri = mesh.triangles
    fmask = np.ones(len(tri), bool) if keep is None else keep(tri.mean(axis=1))
    n = mesh.face_normals
    facing = (n @ d < 0) & fmask
    tri_v, n_v = tri[facing], n[facing]
    depth = tri_v.mean(axis=1) @ d
    dmin, dmax = depth.min(), depth.max()
    near = 1 - (depth - dmin) / (dmax - dmin + 1e-9)          # 1 = closest to the camera
    shade = (0.45 + 0.55 * np.abs(n_v @ d)) * (0.62 + 0.38 * near)
    cols = [tuple(np.clip(BODY * s, 0, 1)) for s in shade]
    polys = [np.stack([t @ right, t @ up], axis=-1) for t in tri_v]
    # sharp feature edges, drawn in depth order with the faces so hidden ones get covered
    ang = mesh.face_adjacency_angles
    fa = mesh.face_adjacency
    sharp = (ang > np.radians(25)) & fmask[fa[:, 0]] & fmask[fa[:, 1]] & (facing[fa[:, 0]] | facing[fa[:, 1]])
    ev = mesh.vertices[mesh.face_adjacency_edges[sharp]]
    edepth = ev.mean(axis=1) @ d - 0.3
    epolys = [np.stack([e @ right, e @ up], axis=-1) for e in ev]
    allp = polys + epolys
    alld = np.concatenate([depth, edepth])
    fc = cols + ["none"] * len(epolys)
    ec = cols + ["#3b4250"] * len(epolys)
    lw = [0.3] * len(polys) + [0.9] * len(epolys)
    order = np.argsort(-alld)
    ax.add_collection(PolyCollection([allp[i] for i in order], facecolors=[fc[i] for i in order],
                                     edgecolors=[ec[i] for i in order], linewidths=[lw[i] for i in order], closed=False))
    xy = np.concatenate(polys)
    ax.set_xlim(xy[:, 0].min() - 18, xy[:, 0].max() + 18)
    ax.set_ylim(xy[:, 1].min() - 18, xy[:, 1].max() + 18)
    ax.set_aspect("equal"); ax.axis("off")
    return lambda p: (np.dot(p, right), np.dot(p, up)), right, up


def mark(ax, proj, p, num, color, arrow_from=None):
    x, y = proj(p)
    if arrow_from is not None:
        ax.annotate("", xy=(x, y), xytext=arrow_from, arrowprops=dict(arrowstyle="-|>", lw=2.6, color=color), zorder=9)
    ax.add_patch(plt.Circle((x, y), 5.2, fc=color, ec="white", lw=1.8, zorder=10))
    ax.text(x, y, str(num), ha="center", va="center", color="white", fontsize=10, weight="bold", zorder=11)


def note(ax, proj, p, text, dx=0, dy=0, **k):
    x, y = proj(p)
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), fontsize=9.5, color="#333", ha="center",
                arrowprops=dict(arrowstyle="-", color="#888", lw=1) if (dx or dy) else None, zorder=12, **k)


def page(title, subtitle, panels, legend, fn, size=(11, 8.5)):
    fig = plt.figure(figsize=size, dpi=140)
    fig.suptitle(title, fontsize=19, weight="bold", y=0.975)
    fig.text(0.5, 0.94, textwrap.fill(subtitle, int(size[0] * 11)), ha="center", va="top", fontsize=11.5, color="#444")
    n = len(panels)
    axes = [fig.add_axes([0.03 + i * 0.94 / n, 0.2, 0.94 / n - 0.02, 0.62]) for i in range(n)]
    for ax, (label, draw) in zip(axes, panels):
        draw(ax)
        ax.set_title(label, fontsize=12, weight="bold", color="#222", pad=6)
    y = 0.155
    for color, text in legend:
        fig.patches.append(plt.Circle((0.06, y + 0.008), 0.009, transform=fig.transFigure, fc=color, ec="none"))
        fig.text(0.08, y, text, fontsize=11, va="center")
        y -= 0.045
    fig.savefig(os.path.join(OUT, fn), facecolor="white")
    plt.close(fig)
    print("wrote", fn)


# ---------------------------------------------------------------- 1. base_frame (4, from underneath)
def base_frame_page():
    m = P.base_frame()

    def draw(ax):
        proj, *_ = render(ax, m, (0, 0, 1), (0, 1, 0))
        for i, (x, y) in enumerate(P.FRAME_SCREWS, 1):
            mark(ax, proj, (x, y, P.FLOOR_T), i, COL[0])
        note(ax, proj, (-100, 0, 20), "cable opening\n(-x side)", dx=-32 * np.sign(proj((1, 0, 0))[0] or 1), dy=0)
    page("base_frame: 4 inserts", "Flip it over so the open skirt faces you. The 4 inserts go into the small blocks inside the skirt, near the corners.",
         [("Looking at the BOTTOM (open side up)", draw)],
         [(COL[0], "1-4: from the bottom, straight down into each block. Later: M3x6 up through base_floor into these.")],
         "1_base_frame_inserts.png")


# ---------------------------------------------------------------- 2. turntable (16 from the top + 3 sideways)
def turntable_page():
    m = P.turntable()
    sleeve = [pt(P.PAN_M, (x, y, -P.SV_FACE)) for x, y in P.INS_XY]
    towers = [(x, sy * yy, P.TT_Z1) for sy in (-1, 1) for x in (-22, 22) for yy in (P.A + 8, P.A + 22)]
    boxes = [(x, y, P.TT_Z1) for x, y in P.EB_SCREWS]
    posts = []
    for ang in P.ROLL_ANG:
        c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        posts.append((np.array((P.ROLL_R * c, P.ROLL_R * s, P.ROLL_Z)), np.array((c, s, 0.0))))

    def top(ax):
        proj, *_ = render(ax, m, (0, 0, -1), (0, 1, 0))
        k = 1
        for p in sleeve: mark(ax, proj, p, k, COL[0]); k += 1
        for p in towers: mark(ax, proj, p, k, COL[1]); k += 1
        for p in boxes: mark(ax, proj, p, k, COL[2]); k += 1
        note(ax, proj, (P.SV_AX, 0, P.TT_Z1), "pan servo\nopening", dy=0)
        note(ax, proj, (-80.5, -29, P.TT_Z1), "cable slot", dx=-6, dy=-24)

    def bottom(ax):
        proj, *_ = render(ax, m, (0, 0, 1), (0, 1, 0))
        for i, (p, r) in enumerate(posts, 17):
            x0, y0 = proj(p + r * 30)
            mark(ax, proj, p, i, COL[3], arrow_from=(x0, y0))
        note(ax, proj, (0, 0, 44), "servo sleeve", dy=0)

    page("turntable: 19 inserts", "Top face (the flat side that was on the print bed) gets 16. The 3 roller posts underneath get 1 each, from the side.",
         [("TOP face (flat side)", top), ("UNDERSIDE (sleeve + 3 posts)", bottom)],
         [(COL[0], "1-4: rim around the servo opening. Later: pan floor cap, 4x M3x12."),
          (COL[1], "5-12: two rows each side, for the towers. Later: 4x M3x10 per tower."),
          (COL[2], "13-16: electronics box corners. Later: M3x20 through tray + box."),
          (COL[3], "17-19: SIDEWAYS into the outer face of each post (arrow). Lay the turntable on its edge so the hole faces up.")],
         "2_turntable_inserts.png", size=(13, 8.8))


# ---------------------------------------------------------------- 3. tower_servo (4 in the sleeve opening)
def tower_servo_page():
    m = P.tower_servo()
    pts = [pt(P.TILT_M, (x, y, -P.SV_FACE)) for x, y in P.INS_XY]

    def draw(ax):
        proj, *_ = render(ax, m, (0, 1, 0), (0, 0, 1))
        for i, p in enumerate(pts, 1):
            mark(ax, proj, p, i, COL[0])
        note(ax, proj, pt(P.TILT_M, (0, 0, -P.SV_FACE)), "servo\nopening", dy=0)
        note(ax, proj, (0, -60, P.TT_Z1 + 3), "foot (screws to the turntable)", dy=-14)

    page("tower_servo: 4 inserts", "Look at the outside face, where the box-shaped servo sleeve sticks out. The 4 inserts go into the sleeve's rim, around the opening.",
         [("OUTSIDE face (sleeve toward you)", draw)],
         [(COL[0], "1-4: into the sleeve rim corners. Later: tilt floor cap, 4x M3x12.")],
         "3_tower_servo_inserts.png", size=(8.5, 10))


# ---------------------------------------------------------------- 4. laser_plate (1 on top + 3 in the ear)
def laser_plate_page():
    m = P.laser_plate()
    ss = (40.0, 0.0, P.CR_Z1 + 22)
    ear = [(x, P.YI, P.ZT + z) for x, z in P.STUB_BOLTS]

    def top(ax):
        proj, *_ = render(ax, m, (0, 0, -1), (0, 1, 0))
        mark(ax, proj, ss, 1, COL[0])
        note(ax, proj, (-39, 0, P.CR_Z1 + 20), "camera riser", dy=0)
        note(ax, proj, (0, P.YI + 2.5, P.ZT), "stub-shaft ear", dx=0, dy=14)

    def inner(ax):
        proj, *_ = render(ax, m, (0, 1, 0), (0, 0, 1), keep=lambda c: c[:, 1] > P.YI - 0.5)
        for i, p in enumerate(ear, 2):
            mark(ax, proj, p, i, COL[1])
        note(ax, proj, (0, P.YI, P.ZT), "shaft hole", dy=0)

    page("laser_plate: 4 inserts", "1 on top of the laser block (above the laser bore), 3 in the ear on the stub-shaft side, from the INSIDE face.",
         [("TOP", top), ("Stub-shaft ear, seen from the middle of the plate", inner)],
         [(COL[0], "1: top of the laser block. Later: an M3 set screw that clamps the laser in its bore (snug, don't crush it)."),
          (COL[1], "2-4: around the shaft hole, inside face. Later: stub shaft flange, 3x M3 into these.")],
         "4_laser_plate_inserts.png", size=(13, 8.2))


# ---------------------------------------------------------------- 5. stub_shaft (1 in the end)
def stub_shaft_page():
    m = P.stub_shaft()
    p = (0.0, P.A + 6.5, P.ZT)

    def draw(ax):
        proj, *_ = render(ax, m, (0, -1, 0), (0, 0, 1))
        mark(ax, proj, p, 1, COL[0])
        note(ax, proj, (0, 0, P.ZT - 15), "flange behind", dy=-8)

    page("stub_shaft: 1 insert", "Into the very end of the shaft (the thin end, opposite the flange). Go slowly and keep it centered: the shaft is only 14.9 mm wide.",
         [("Looking at the shaft END", draw)],
         [(COL[0], "1: shaft tip, straight in along the shaft. Later: M3 + washer that keeps the bearing on.")],
         "5_stub_shaft_inserts.png", size=(8.5, 8.5))


if __name__ == "__main__":
    base_frame_page(); turntable_page(); tower_servo_page(); laser_plate_page(); stub_shaft_page()
