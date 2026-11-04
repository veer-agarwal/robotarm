"""Electronics move-in: ebox inserts + servo board, ebox + tray onto the turntable, and the wiring (laser phase)."""
import os
os.environ.setdefault("TURRET_PAYLOAD", "laser")
import textwrap
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import perses_turret as P
from assembly_map import (crop, render_multi, tag, path, page, screw_mesh, screws_named,
                          C_PRINT, C_SERVO, C_SCREW)
from insert_map import render, mark

OUT = os.path.join(P.HERE, "out", "assembly")
C = lambda h: np.array(matplotlib.colors.to_rgb(h))
Z = np.array((0, 0, 1.0))
C_BOX, C_TRAY, C_LID = C("#4f9d6c"), C("#8fd0a6"), C("#6cc08b")
C_BOARD, C_TIE = C("#2050c0"), C("#f2f2f2")
bxr, byr, bzr = P.BOARD_BOX


def bx(lo, hi):
    lo, hi = np.array(lo, float), np.array(hi, float)
    b = trimesh.creation.box(extents=hi - lo); b.apply_translation((lo + hi) / 2); return b


def tube(pts, r=1.8):
    pts = [np.asarray(p, float) for p in pts]
    segs = [trimesh.creation.cylinder(radius=r, segment=np.array([a, b]), sections=10) for a, b in zip(pts[:-1], pts[1:]) if np.linalg.norm(b - a) > 0.5]
    return trimesh.util.concatenate(segs + [trimesh.creation.icosphere(subdivisions=1, radius=r).apply_translation(p) for p in pts])


BOARD = bx((bxr[0], byr[0] + 0.2, bzr[0]), (bxr[1], byr[0] + 2.0, bzr[1]))          # PCB against the wall
PARTS = bx((bxr[0] + 3, byr[0] + 2.0, bzr[0] + 3), (bxr[1] - 3, byr[0] + 9, bzr[1] - 6))  # jacks / headers
TIES = [bx((22.5, byr[0] + 9, z + 0.8), (71.5, byr[0] + 10.2, z + 3.2)) for z in (122.0, 140.0)]


# ---------------------------------------------------------------- 11. prep: 4 lid inserts + servo board inside the box
def prep():
    fig = plt.figure(figsize=(14, 9.8), dpi=140)
    fig.suptitle("ebox: 4 lid inserts, then the servo board goes inside", fontsize=19, weight="bold", y=0.975)

    ax1 = fig.add_axes([0.0, 0.3, 0.42, 0.6])
    proj, *_ = render(ax1, P.parts["ebox"], (0, 0, -1), (0, 1, 0))
    for k, (x, y) in enumerate(P.EB_SCREWS, 1):
        mark(ax1, proj, (x, y, P.EB_Z1), k, "#17a34a")
    x0, y0 = proj((50, -P.EB_Y, P.EB_Z1)); ax1.text(x0, y0 - 9, "notch: tilt lead comes in here\n(this wall faces tower_servo)", ha="center", va="top", fontsize=9.5, weight="bold", color="#1d4f91")
    x0, y0 = proj((30, P.EB_Y, P.EB_Z1)); ax1.text(x0, y0 + 8, "vent slots (faces tower_bearing)", ha="center", va="bottom", fontsize=9.5, color="#444")
    ax1.set_title("Open top, seen from above: 4 inserts, M3 H5", fontsize=12, weight="bold")

    ax2 = fig.add_axes([0.42, 0.3, 0.58, 0.6])
    shell = crop(P.parts["ebox"], ymax=10)
    proj2 = render_multi(ax2, [(shell, C_BOX), (BOARD, C("#1b6b30")), (PARTS, C_BOARD)] + [(t, C_TIE) for t in TIES], d=(0.25, -0.85, -0.45))
    tag(ax2, proj2, np.array(((bxr[0] + bxr[1]) / 2, byr[0] + 9, bzr[1] - 8)), "servo board, ports UP,\nparts facing into the box", -45, -50)
    tag(ax2, proj2, np.array((71.5, byr[0] + 10, 141.5)), "2 zip ties through\nthe 4 small slots", 45, -25)
    tag(ax2, proj2, np.array((P.EB_X0, -30, P.TRAY_Z1 + 8)), "low opening: 12 V + USB out\n(back of the turret)", -15, -35)
    ax2.set_title("Box cut open so you can see the inside wall", fontsize=12, weight="bold")

    steps = ["Heat-set 4 M3 H5 inserts into the 4 corner bosses at the box's OPEN top (green 1-4). These hold the lid later. Same method as before: iron on, let it sink flush.",
             "Peel the servo board off tower_servo's inner face. The box's wall sits right where the board is now, so it has to move.",
             "Find the wall with the notch at the top and 4 small slots (not the one with the 7 tall vents). Stand the board against it on the INSIDE, ports UP, top edge just under the notch.",
             "2 zip ties: in one slot, across the front of the board, out the other slot at the same height. Snug, then snip. No zip ties: double-sided tape between the slots works too.",
             "Leave the 12 V and USB plugged into the board. You'll feed them out the low rectangular opening in the next step."]
    txt = "\n".join(textwrap.fill(f"{i}. {s}", 160, subsequent_indent="    ") for i, s in enumerate(steps, 1))
    fig.text(0.05, 0.255, txt, fontsize=11.5, va="top", linespacing=1.55)
    fig.savefig(os.path.join(OUT, "11_ebox_prep.png"), facecolor="white"); plt.close(fig); print("wrote 11_ebox_prep.png")


# ---------------------------------------------------------------- 12. box + tray onto the turntable
def install():
    tt = trimesh.util.concatenate([P.turntable(), P.parts["pan_floor_cap"], P.parts["tower_servo"], P.parts["tilt_floor_cap"]])
    sv = trimesh.util.concatenate([P.hardware["servo_pan"], P.hardware["servo_tilt"]])
    up_box, up_tray, up_sc = 45, 100, 150
    box = P.parts["ebox"].copy(); box.apply_translation(Z * up_box)
    brd = trimesh.util.concatenate([BOARD, PARTS] + TIES); brd.apply_translation(Z * up_box)
    tray = P.parts["ebox_tray"].copy(); tray.apply_translation(Z * up_tray)
    sc = [screw_mesh(h + Z * up_sc, a, L) for h, a, L in screws_named("tray + box")]
    pan_exit = (P.PAN_M @ [-2.0, 0, -P.SV_FACE - P.FCAP_T, 1])[:3]
    lead = tube([pan_exit, pan_exit + Z * 10, (14, 0, P.TRAY_Z1 + up_tray - 2), (14, 0, P.TRAY_Z1 + up_tray + 14)], 1.6)
    seat = [np.array((x, y, P.TT_Z1)) for x, y in P.EB_SCREWS]

    def draw(ax):
        proj = render_multi(ax, [(tt, C_PRINT), (sv, C_SERVO), (box, C_BOX), (brd, C_BOARD), (tray, C_TRAY), (lead, C("#7b2cbf"))] +
                            [(m, C_SCREW) for m in sc], d=(0.5, -0.62, -0.6))
        for s in seat:
            path(ax, proj, s + Z * (up_sc + P.TRAY_Z1 - P.TT_Z1 - 4), s + Z * 2)
        tag(ax, proj, seat[1] + Z * (up_sc + 20), "4x M3x20 through tray + box\ninto turntable inserts 13-16", 80, 40)
        tag(ax, proj, np.array((P.XI1, 0, P.TRAY_Z1 + up_tray)), "ebox_tray (drops in onto\nthe 4 corner seats)", 130, 10)
        tag(ax, proj, np.array((P.EB_X1, P.EB_Y, 120 + up_box)), "ebox (notch wall goes\ntoward tower_servo)", 120, 0)
        tag(ax, proj, np.array((P.EB_X0, -30, P.TRAY_Z1 + 6 + up_box)), "low opening = BACK\n(cable-slot side)", -110, -10)
        tag(ax, proj, np.array((14, 0, P.TRAY_Z1 + up_tray + 14)), "pan lead up through\nthe tray's square hole", -110, 60, color="#5a1f8f")
        tag(ax, proj, np.array((-20, -P.A - 20, P.TT_Z1 + 60)), "tower_servo", 100, 70)

    page("ebox + tray onto the turntable (tower_bearing hidden)",
         ["Unplug the pan and tilt leads from the board. Lower the box between the towers: notch wall toward tower_servo, the low opening toward the BACK (cable-slot side), "
          "the 2 round holes facing forward with the laser. Its 4 corner seats land on turntable inserts 13-16.",
          "Push the pan servo lead up through the tray's square hole, then drop the tray into the box onto the 4 corner seats (rails side up).",
          "4x M3x20 down through the tray and the seats into inserts 13-16. Snug, not tight.",
          "Feed the 12 V + USB out through the low opening in the back wall, then down the turntable's cable slot like before.",
          "The 2 round holes on the front are for a future arm switch: leave them empty for now."],
         draw, "12_ebox_on_turntable.png", size=(13, 10))


# ---------------------------------------------------------------- 13. wiring, top view
def wiring():
    fig = plt.figure(figsize=(13, 12), dpi=140)
    fig.suptitle("Wiring with the ebox (top view, lid off)", fontsize=19, weight="bold", y=0.975)
    ax = fig.add_axes([0.03, 0.3, 0.62, 0.63])
    items = [P.turntable(), P.parts["tower_servo"], P.parts["tower_bearing"], P.parts["ebox"], P.parts["ebox_tray"]]
    render(ax, trimesh.util.concatenate([m.copy() for m in items]), (0, 0, -1), (0, 1, 0))
    ax.set_xlim(-118, 118); ax.set_ylim(-112, 112)

    def box(x0, y0, x1, y1, fc, ec, label=None, tc="#111", z=8, a=0.95):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=1.8, alpha=a, zorder=z))
        if label: ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center", fontsize=9, weight="bold", color=tc, zorder=z + 1)

    def wire(pts, color, label, ls="-"):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=3.2, ls=ls, solid_capstyle="round", zorder=11, label=label)
        ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5), zorder=11)

    box(bxr[0], byr[0], bxr[1], byr[0] + 6, "#2050c0", "#0d2a6b", None, z=9, a=1)
    ax.annotate("servo board\n(on the inside wall)", xy=(47, byr[0] + 3), xytext=(47, -14), ha="center", fontsize=9.5, weight="bold",
                color="#0d2a6b", arrowprops=dict(arrowstyle="-", color="#0d2a6b"), zorder=13, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#0d2a6b"))
    box(-76, -41, -22, 41, "#fafafa", "#666", None, z=8)
    box(-64, -14, -34, 14, "#222", "#000", "ESP32", tc="white", z=9, a=1)
    ax.text(-49, 30, "breadboard\n(stays OUTSIDE,\nbehind the box)", ha="center", fontsize=8.5, weight="bold", zorder=10)
    box(-84, -33, -77, -25, "#d62728", "#7a0000", None, z=9, a=1)
    ax.annotate("cable slot", xy=(-80.5, -29), xytext=(-104, -70), fontsize=9.5, weight="bold", color="#7a0000", arrowprops=dict(arrowstyle="-", color="#7a0000"), zorder=13)
    ax.annotate("", xy=(112, 0), xytext=(86, 0), arrowprops=dict(arrowstyle="-|>", lw=3, color="#444"))
    ax.text(99, 8, "laser\nforward", ha="center", fontsize=9, color="#444")

    wire([(14.5, -2), (14, -12), (28, -30), (34, byr[0] + 6)], "#7b2cbf", "pan lead: up through the tray's square hole")
    wire([(22, -P.A - 34), (43, -P.A - 6), (43, byr[0] + 6)], "#1f77b4", "tilt lead: in through the notch at the top of the wall")
    wire([(60, byr[0] + 5), (-12, -32), (-21, -30), (-79, -30)], "#2ca02c", "12 V + USB: out the low back opening, into the slot")
    wire([(-60, -14), (-70, -25), (-80, -30)], "#555", "ESP32 USB: into the slot (as before)", ls="--")
    wire([(30, 4), (-10, 6), (-22, 6), (-36, 6)], "#d62728", "laser leads: over the back edge of the lid, with slack")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10, frameon=False)

    notes = ("Why the breadboard stays outside: the tray has the old Nano rails molded in, so a full breadboard doesn't fit on it.\n"
             "It sits where it is now, behind the box. Slide it ~3 mm further back so it clears the box's back wall.\n\n"
             "Laser leads: leave a slack loop where they leave the laser plate, so the plate can tilt its full range without\n"
             "pulling them. The plate clears the lid by 25 mm at any tilt.\n\n"
             "Test before the lid: turret_teleop.bat --port COM9 --laser COM11. Pan and tilt to every limit and watch\n"
             "that no lead pulls tight. Then the lid: 4x M3x8 into the new inserts. Every lead has a gap to get out:\n"
             "tilt through the notch (the lid has a matching cut-out), 12 V + USB through the back opening.")
    fig.text(0.05, 0.27, notes, fontsize=11, va="top", linespacing=1.5)
    fig.savefig(os.path.join(OUT, "13_ebox_wiring.png"), facecolor="white"); plt.close(fig); print("wrote 13_ebox_wiring.png")


if __name__ == "__main__":
    prep(); install(); wiring()
