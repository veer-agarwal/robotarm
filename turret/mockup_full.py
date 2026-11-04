"""Full turret mockup: Perses in the cradle, laser + webcam on the cradle bracket, electronics in the box,
every cable drawn as a tube along its real route. Run with TURRET_PAYLOAD=perses."""
import os
os.environ["TURRET_PAYLOAD"] = "perses"
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trimesh.transformations import rotation_matrix as Rot
import perses_turret as P
from assembly_map import render_multi, crop

OUT = os.path.join(P.HERE, "out", "assembly")
C = lambda h: np.array(matplotlib.colors.to_rgb(h))
COL = {"print": C("#c3ccd8"), "tower": C("#5b8fd0"), "cradle": C("#e8a33d"), "perses": C("#e05a4f"), "dark": C("#2a2d33"),
       "box": C("#4f9d6c"), "lid": C("#6cc08b"), "board": C("#2050c0"), "bread": C("#f2f2f2"), "esp": C("#151515"),
       "brg": C("#c9a227"), "laser": C("#b22222"), "cam": C("#202020")}
WCOL = {"pan": "#7b2cbf", "tilt": "#1f77b4", "laser": "#d62728", "cam": "#555555", "power": "#2ca02c"}


def tube(pts, r=1.8):
    pts = [np.asarray(p, float) for p in pts]
    segs = [trimesh.creation.cylinder(radius=r, segment=np.array([a, b]), sections=10) for a, b in zip(pts[:-1], pts[1:]) if np.linalg.norm(b - a) > 0.5]
    joints = [trimesh.creation.icosphere(subdivisions=1, radius=r).apply_translation(p) for p in pts]
    return trimesh.util.concatenate(segs + joints)


def bx(lo, hi):
    lo, hi = np.array(lo, float), np.array(hi, float)
    b = trimesh.creation.box(extents=hi - lo); b.apply_translation((lo + hi) / 2); return b


def build(tilt=0.0, lid=True, perses=True, bare=False):
    Mt = Rot(np.radians(tilt), (0, 1, 0), point=(0, 0, P.ZT))
    items = []
    tower_names = {"tower_servo", "tower_bearing"}
    for n, m in P.parts.items():
        if n == "ebox_lid" and not lid: continue
        if bare and (n in P.TILTING or n == "tower_bearing"): continue
        mm = m.copy()
        if n in P.TILTING: mm.apply_transform(Mt)
        col = (COL["tower"] if n in tower_names else COL["cradle"] if n == "cradle" else COL["box"] if n in ("ebox", "ebox_tray")
               else COL["lid"] if n == "ebox_lid" else COL["print"])
        items.append((mm, col))
    for n, m in P.hardware.items():
        if n in ("arduino_nano", "usb_hub", "future_pi5"): continue
        if n == "perses" and not perses: continue
        if bare and (n in P.TILTING or n == "bearing_tilt"): continue
        mm = m.copy()
        if n in P.TILTING: mm.apply_transform(Mt)
        col = (COL["perses"] if n == "perses" else COL["board"] if n == "servo_board" else COL["brg"] if "bearing" in n else COL["dark"])
        items.append((mm, col))

    # ESP32 on a half breadboard, on the box tray (replaces the Nano + hub)
    z0 = P.TRAY_Z1 + 10.5                                   # sits on the old Nano rails
    items.append((bx((15, -14, z0), (70, 41, z0 + 9)), COL["bread"]))
    items.append((bx((24, 0, z0 + 9), (52, 30, z0 + 13)), COL["esp"]))

    # laser + webcam on the cradle's front bracket (tilt with the cradle)
    zt = P.CR_Z1 + P.CR_WALL_H
    las = trimesh.creation.cylinder(radius=5.25, segment=np.array([(P.CR_LEN / 2 - 30, -P.YO - 21, zt - 14), (P.CR_LEN / 2 + 1, -P.YO - 21, zt - 14)]))
    cam = bx((P.CR_LEN / 2 - 66, -P.YO - 29, zt), (P.CR_LEN / 2 + 28, -P.YO - 1, zt + 30))
    for m, c in ((las, COL["laser"]), (cam, COL["cam"])):
        m.apply_transform(Mt)
        if not bare: items.append((m, c))

    # ---- cables
    bxr, byr, bzr = P.BOARD_BOX
    port = (bxr[0] + 6, byr[1] + 3, bzr[1] - 3)
    pan_exit = (P.PAN_M @ [-2.0, 0, -P.SV_FACE - P.FCAP_T, 1])[:3]
    tilt_exit = (P.TILT_M @ [-2.0, 0, -P.SV_FACE - P.FCAP_T, 1])[:3]
    wires = {
        "pan": [pan_exit, pan_exit + (0, 0, 9), (6, -8, 99), (6, -20, 103), (10, -32, 118), (24, -35, 130), port],
        "tilt": [tilt_exit, tilt_exit + (0, -2.5, 0), (22, tilt_exit[1] - 2.5, tilt_exit[2] - 5), (34, -48, P.EB_Z1 + 8),
                 (40, -P.EB_Y + 1, P.EB_Z1 - 2), (40, byr[1] + 3, bzr[1] - 3)],
    }
    # laser + webcam leads: from the bracket, down the outside of the cradle wall, a slack loop near the tilt axis, into the lid
    def tilted(p): return (Mt @ np.r_[p, 1])[:3]
    las_back = tilted((P.CR_LEN / 2 - 31, -P.YO - 21, zt - 14))
    cam_back = tilted((P.CR_LEN / 2 - 66, -P.YO - 15, zt + 12))
    wires["laser"] = [las_back, tilted((60, -P.YO - 12, P.CR_Z0 - 4)), tilted((30, -P.YO - 6, P.ZT + 6)),
                      (26, -P.YO + 6, P.ZT - 30), (40, -14, P.EB_Z1 + 18), (40, -8, P.EB_Z1 - 4), (38, 12, z0 + 14)]
    wires["cam"] = [cam_back, tilted((50, -P.YO - 8, P.CR_Z0 - 2)), tilted((24, -P.YO - 2, P.ZT + 10)),
                    (20, -P.YO + 10, P.ZT - 34), (30, -10, P.EB_Z1 + 22), (30, -4, P.EB_Z1 - 4), (30, 10, z0 + 16)]
    # 12 V + USB: box -x wall opening -> turntable slot -> loop in the cable tray -> base cable opening
    loop = [(89 * np.cos(a), 89 * np.sin(a), P.DECK_TOP + 6) for a in np.radians(np.linspace(-160, 160, 25))]
    wires["power"] = [(P.EB_X0 + 3, -30, P.TRAY_Z1 + 6), (P.EB_X0 - 12, -30, P.TT_Z1 + 3), (-80.5, -29, P.TT_Z1 + 2),
                      (-80.5, -29, P.TT_Z0 - 4), (-84, -31, P.DECK_TOP + 6)] + loop[::-1][::1] + [(-92, 0, 30), (-110, 0, 20), (-150, 0, 15)]
    tubes = {k: tube(v, 2.2 if k == "power" else 1.7) for k, v in wires.items()}
    for k, t in tubes.items():
        if bare and k in ("laser", "cam"):
            t = crop(t, zmax=P.EB_Z1 + 25)
        items.append((t, C(WCOL[k])))
    return items


def legend(fig, x, y):
    labels = [("pan servo lead", "pan"), ("tilt servo lead", "tilt"), ("laser leads", "laser"), ("webcam USB", "cam"),
              ("12 V + USB to the PC (loop in the cable tray)", "power")]
    for i, (t, k) in enumerate(labels):
        fig.lines.append(matplotlib.lines.Line2D([x, x + 0.025], [y - i * 0.028] * 2, transform=fig.transFigure, color=WCOL[k], lw=4))
        fig.text(x + 0.032, y - i * 0.028, t, va="center", fontsize=10.5)


def main():
    fig = plt.figure(figsize=(16, 11.5), dpi=130)
    fig.suptitle("Turret, fully built: Perses in the cradle, laser + webcam, electronics in the box", fontsize=19, weight="bold", y=0.985)

    ax1 = fig.add_axes([0.0, 0.32, 0.62, 0.62])
    render_multi(ax1, build(0, lid=True), d=(-0.62, 0.55, -0.56))
    ax1.set_title("Level, lid on", fontsize=12, weight="bold")

    ax2 = fig.add_axes([0.6, 0.47, 0.4, 0.47])
    render_multi(ax2, build(20, lid=True), d=(0.0, 1.0, -0.08))
    ax2.set_title("Side view, tilted 20 deg (Perses limit)", fontsize=12, weight="bold")

    # box open close-up: crop everything to the box region, lid off, no Perses
    items = []
    for m, c in build(0, lid=False, perses=False, bare=True):
        try:
            mm = crop(m, zmax=200)
        except Exception:
            mm = m
        if len(mm.faces): items.append((mm, c))
    ax3 = fig.add_axes([0.02, 0.0, 0.5, 0.36])
    render_multi(ax3, items, d=(-0.25, -0.62, -0.74))
    ax3.set_title("Lid off, cradle + bearing tower hidden: servo board (blue) on the box wall, ESP32 on the tray", fontsize=11, weight="bold")

    legend(fig, 0.6, 0.38)
    fig.text(0.6, 0.2, "Only 12 V + USB cross the pan joint: they drop through the turntable's slot\n"
             "and loop around the cable tray, so the turret pans +-120 deg without pulling.\n"
             "Laser and webcam leads hang in a slack loop near the tilt axis,\n"
             "so the cradle tilts +-20 deg freely. Servo leads are the stock 152 mm.\n"
             "(Perses shape estimated from photos: measure the real grip before printing the cradle/ebox.)",
             fontsize=10.5, va="top", linespacing=1.5)
    fig.savefig(os.path.join(OUT, "9_full_turret_mockup.png"), facecolor="white")
    print("wrote 9_full_turret_mockup.png")


if __name__ == "__main__":
    main()
