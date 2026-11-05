"""Ebox install, beginner version: a parts guide with plain names, then one picture per step.
Laser plate stays on, so the box slides in from the front (checked for collisions), tray and lid go in over the top."""
import os
os.environ.setdefault("TURRET_PAYLOAD", "laser")
import textwrap
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import perses_turret as P
from assembly_map import crop, render_multi, tag, path, screw_mesh, screws_named
from insert_map import render, mark

OUT = os.path.join(P.HERE, "out", "assembly", "ebox")
os.makedirs(OUT, exist_ok=True)
C = lambda h: np.array(matplotlib.colors.to_rgb(h))
COL = {"turret": C("#c7ccd4"), "towerS": C("#8fb3e0"), "towerB": C("#b9c7da"), "plate": C("#e0b25a"),
       "box": C("#3fa864"), "tray": C("#ef8a3c"), "lid": C("#9b6fd1"), "board": C("#1f5fd6"), "servo": C("#33363d"),
       "screw": C("#2b2d33"), "cap": C("#d9a0a0"), "bread": C("#f4f4f4")}
HEX = {k: matplotlib.colors.to_hex(v) for k, v in COL.items()}
Z, X = np.array((0, 0, 1.0)), np.array((1.0, 0, 0))
bxr, byr, bzr = P.BOARD_BOX


def bx(lo, hi):
    lo, hi = np.array(lo, float), np.array(hi, float)
    b = trimesh.creation.box(extents=hi - lo); b.apply_translation((lo + hi) / 2); return b


BOARD = trimesh.util.concatenate([bx((bxr[0], byr[0] + 0.2, bzr[0]), (bxr[1], byr[0] + 2.0, bzr[1])),
                                  bx((bxr[0] + 3, byr[0] + 2.0, bzr[0] + 3), (bxr[1] - 3, byr[0] + 9, bzr[1] - 6))])
BREAD = bx((-76, -41, P.TT_Z1), (-22, 41, P.TT_Z1 + 9))
D_VIEW = (-0.55, -0.5, -0.67)          # camera above the front-right (laser end + bearing tower side)


def turret(plate=True):
    items = [(P.turntable(), COL["turret"]), (P.parts["pan_floor_cap"], COL["cap"]),
             (P.parts["tower_servo"], COL["towerS"]), (P.parts["tilt_floor_cap"], COL["towerS"]),
             (P.hardware["servo_tilt"], COL["servo"]), (P.parts["tower_bearing"], COL["towerB"]), (BREAD, COL["bread"])]
    if plate:
        items += [(P.parts["laser_plate"], COL["plate"]), (P.parts["horn_spacer"], COL["plate"]),
                  (P.hardware["laser"], C("#b22222"))]
    return items


def moved(m, v):
    m = m.copy(); m.apply_translation(v); return m


def step_page(num, title, steps, draw, fn, size=(13, 10.5)):
    fig = plt.figure(figsize=size, dpi=140)
    fig.suptitle(f"Step {num}: {title}", fontsize=20, weight="bold", y=0.975)
    ax = fig.add_axes([0.02, 0.29, 0.96, 0.63])
    draw(ax)
    txt = "\n".join(textwrap.fill(f"{chr(97 + i)}) {s}", 135, subsequent_indent="    ") for i, s in enumerate(steps))
    fig.text(0.05, 0.255, txt, fontsize=12.5, va="top", linespacing=1.6)
    fig.savefig(os.path.join(OUT, fn), facecolor="white"); plt.close(fig); print("wrote", fn)


def key(ax, names):
    """Small colour key in the corner."""
    for i, (label, k) in enumerate(names):
        ax.add_patch(plt.Rectangle((0.0, 0.97 - i * 0.045), 0.025, 0.03, transform=ax.transAxes, fc=HEX[k], ec="#333", lw=0.6, zorder=30))
        ax.text(0.035, 0.985 - i * 0.045, label, transform=ax.transAxes, fontsize=11, va="center", zorder=30)


# ------------------------------------------------------------------ 0. parts guide
def parts():
    fig = plt.figure(figsize=(14, 12), dpi=140)
    fig.suptitle("Know your parts first (colours match every step picture)", fontsize=20, weight="bold", y=0.98)
    ax = fig.add_axes([0.05, 0.42, 0.53, 0.48])
    proj = render_multi(ax, turret(), d=D_VIEW)
    tag(ax, proj, np.array((0, -P.A - 25, 200)), "TILT-SERVO TOWER\n(the black servo sticks out of it)", -95, 55, color="#1d4f91")
    tag(ax, proj, np.array((0, P.A + 15, 215)), "BEARING TOWER\n(round bearing, no servo)", 60, 75)
    tag(ax, proj, np.array((50, 0, 240)), "LASER PLATE\n(stays on, don't remove)", 130, 10, color="#8a5a00")
    tag(ax, proj, np.array((12, 0, 88)), "PAN SERVO CAP\n(pink block in the middle)", 140, -60, color="#a33")
    tag(ax, proj, np.array((-49, 0, 89)), "BREADBOARD + ESP32", -120, -40)
    tag(ax, proj, np.array((90, 0, 80)), "FRONT = laser end", 40, -55)
    ax.set_title("What's already on your turret", fontsize=13, weight="bold")

    gal = [("BOX", "ebox", "box", "Biggest new piece. 4 walls,\nopen top AND bottom.\nOne wall: 7 tall slots.\nOpposite wall: a notch at\nthe top + 4 small slots."),
           ("TRAY", "ebox_tray", "tray", "Flat plate that sits inside the box.\nHas a SQUARE HOLE, 2 raised rails\nand 4 corner screw holes."),
           ("LID", "ebox_lid", "lid", "Thin cover with a lip underneath,\n2 round holes and a cut-out on one edge.")]
    for i, (name, part, k, desc) in enumerate(gal):
        a = fig.add_axes([0.6, 0.67 - i * 0.24, 0.17, 0.22])
        render_multi(a, [(P.parts[part], COL[k])], d=(-0.5, -0.55, -0.67))
        fig.text(0.78, 0.8 - i * 0.24, name, fontsize=15, weight="bold", color=HEX[k], va="center")
        fig.text(0.78, 0.76 - i * 0.24, desc, fontsize=10.5, va="top")
    a = fig.add_axes([0.62, 0.02, 0.13, 0.13]); a.axis("off"); a.set_xlim(0, 42); a.set_ylim(0, 33)
    a.add_patch(plt.Rectangle((0, 0), 42, 33, fc=HEX["board"], ec="#0d2a6b", lw=2))
    for i in range(6):
        a.add_patch(plt.Rectangle((3 + i * 6.3, 26), 4.5, 6, fc="#111", ec="#000"))
    a.add_patch(plt.Rectangle((2, 4), 9, 9, fc="#222")); a.add_patch(plt.Rectangle((30, 3), 10, 7, fc="#bbb"))
    a.text(21, 22, "servo plugs here (top edge)", ha="center", fontsize=7.5, color="white")
    fig.text(0.78, 0.13, "SERVO BOARD", fontsize=15, weight="bold", color=HEX["board"])
    fig.text(0.78, 0.09, "The blue board the 2 servo plugs\nand the 12 V + USB go into.\nNow taped to the tilt-servo tower.", fontsize=10.5, va="top")

    hw = ("Hardware for the whole job:\n"
          "  4x heat-set inserts M3 (the short H5 ones)  -> box top corners\n"
          "  4x M3x20 screws (longest)  -> tray + box into the turntable\n"
          "  4x M3x8 screws (short)  -> lid\n"
          "  2 zip ties or double-sided tape  -> servo board\n"
          "  Tools: soldering iron (inserts), L-shaped hex key (the screws under the laser plate)")
    fig.text(0.04, 0.36, hw, fontsize=12.5, va="top", family="monospace", linespacing=1.6)
    fig.savefig(os.path.join(OUT, "0_parts_guide.png"), facecolor="white"); plt.close(fig); print("wrote 0_parts_guide.png")


# ------------------------------------------------------------------ 1. inserts
def step1():
    def draw(ax):
        proj, *_ = render(ax, P.parts["ebox"], (0, 0, -1), (0, 1, 0))
        for k, (x, y) in enumerate(P.EB_SCREWS, 1):
            mark(ax, proj, (x, y, P.EB_Z1), k, "#e4572e")
        x, y = proj((43, -P.EB_Y, P.EB_Z1)); ax.annotate("notch (small dip in the top edge)", xy=(x, y), xytext=(x, y - 22), ha="center",
                                                          fontsize=11, weight="bold", color="#1d4f91", arrowprops=dict(arrowstyle="-|>", color="#1d4f91"))
        x, y = proj((30, P.EB_Y, P.EB_Z1)); ax.text(x, y + 10, "wall with 7 tall slots", ha="center", fontsize=11, color="#444")
        ax.set_title("BOX, open top facing up, looking straight down", fontsize=13, weight="bold")
    step_page(1, "4 inserts in the box's top corners", [
        "Stand the BOX with its open top up. In each of the 4 corners there's a round boss with a hole (red 1-4).",
        "Heat-set one M3 H5 insert into each, same as your other inserts: iron on top, let it sink until flush, keep it straight.",
        "Let them cool a minute. These hold the LID at the very end."], draw, "1_inserts.png")


# ------------------------------------------------------------------ 2. board into the box
def step2():
    def draw(ax):
        shell = crop(P.parts["ebox"], ymax=10)
        proj = render_multi(ax, [(shell, COL["box"]), (BOARD, COL["board"])], d=(0.25, -0.85, -0.45))
        tag(ax, proj, np.array(((bxr[0] + bxr[1]) / 2, byr[0] + 9, (bzr[0] + bzr[1]) / 2)), "SERVO BOARD here, plugs pointing UP,\nparts facing into the box", -10, -42, color=HEX["board"])
        tag(ax, proj, np.array((24, byr[0] + 1, 124)), "small slot (zip tie)", -28, -22)
        tag(ax, proj, np.array((70, byr[0] + 1, 142)), "small slot (zip tie)", 38, -8)
        tag(ax, proj, np.array((43, -P.EB_Y, P.EB_Z1 - 3)), "notch", 42, 6)
        tag(ax, proj, np.array((P.EB_X0, -30, P.TRAY_Z1 + 8)), "low rectangular opening\n= the BACK of the box", -12, -32)
        ax.set_title("BOX cut in half so you can see the inside of the notch wall", fontsize=13, weight="bold")
    step_page(2, "move the servo board into the box", [
        "Unplug the 2 servo plugs (pan and tilt) from the SERVO BOARD. Leave the 12 V and USB plugged in.",
        "Peel the board off the TILT-SERVO TOWER.",
        "Inside the BOX, put it against the wall with the NOTCH (not the 7-slot wall): plugs up, top edge just under the notch.",
        "Hold it with 2 zip ties: in one small slot, across the front of the board, out the slot at the same height. Or double-sided tape.",
        "Feed the 12 V and USB cables out through the LOW RECTANGULAR OPENING. That opening is the back of the box."], draw, "2_board_in_box.png")


# ------------------------------------------------------------------ 3. slide the box in
def step3():
    lift, out = 10.0, 150.0
    box = trimesh.util.concatenate([moved(P.parts["ebox"], (out, 0, lift)), moved(BOARD, (out, 0, lift))])
    ghost = P.parts["ebox"]

    def draw(ax):
        proj = render_multi(ax, turret() + [(box, COL["box"])], d=D_VIEW)
        path(ax, proj, np.array((out - 50, 0, 130 + lift)), np.array((20, 0, 130 + lift)))
        tag(ax, proj, np.array((out + 80, 0, 150 + lift)), "BOX: slide it in from the FRONT,\nheld about 1 cm up", 60, 60, color="#1e7a3e")
        tag(ax, proj, np.array((out + 30, -P.EB_Y, 120 + lift)), "notch wall faces the\nTILT-SERVO TOWER side", -150, 20, color="#1d4f91")
        tag(ax, proj, np.array((12, 0, 88)), "PAN SERVO CAP: keep the box\n1 cm up so it slides over this", -40, -90, color="#a33")
        tag(ax, proj, np.array((50, 0, 240)), "LASER PLATE stays on.\nThe box fits under it.", -60, 70, color="#8a5a00")
        ax.set_title("View from above the front (laser end)", fontsize=13, weight="bold")
    step_page(3, "slide the box in under the laser plate", [
        "First slide the BREADBOARD about 3 mm toward the back so it clears the box's back wall.",
        "Hold the BOX by its sides, notch wall toward the TILT-SERVO TOWER, low opening toward the back (where the cables go).",
        "Slide it in from the FRONT (the laser end), between the two towers, about 1 cm above the turntable so it clears the PAN SERVO CAP.",
        "Keep sliding until it's centred, then lower it. Its 4 inside corner posts land on 4 threaded inserts in the turntable.",
        "The 12 V + USB come out the back opening. The tilt servo's plug comes in over the notch."], draw, "3_slide_box_in.png")


# ------------------------------------------------------------------ 4. tray + screws
def step4():
    lift = 70.0
    tray = moved(P.parts["ebox_tray"], (0, 0, lift))
    sc = [screw_mesh(h + Z * (lift + 40), a, L) for h, a, L in screws_named("tray + box")]
    box = trimesh.util.concatenate([P.parts["ebox"], BOARD])
    pan_exit = (P.PAN_M @ [-2.0, 0, -P.SV_FACE - P.FCAP_T, 1])[:3]
    lead = trimesh.util.concatenate([trimesh.creation.cylinder(radius=1.8, segment=np.array([a, b]))
                                     for a, b in ((pan_exit, pan_exit + Z * 10), (pan_exit + Z * 10, (14, 0, P.TRAY_Z1 + lift + 6)))])

    def draw(ax):
        proj = render_multi(ax, turret(plate=False) + [(box, COL["box"]), (tray, COL["tray"]), (lead, C("#7b2cbf"))] + [(m, COL["screw"]) for m in sc], d=D_VIEW)
        for x, y in P.EB_SCREWS:
            path(ax, proj, np.array((x, y, P.TRAY_Z1 + lift + 30)), np.array((x, y, P.TRAY_Z1 + 2)))
        tag(ax, proj, np.array((P.XI1, 0, P.TRAY_Z1 + lift)), "TRAY: slide it over the box top,\nthen drop it inside", 140, 90, color="#c0601a")
        tag(ax, proj, np.array((14, 0, P.TRAY_Z1 + lift + 6)), "pan servo plug comes UP\nthrough the tray's square hole", -150, 70, color="#5a1f8f")
        tag(ax, proj, np.array((73.5, -37.5, P.TRAY_Z1 + lift + 40)), "4x M3x20 (longest screws)", 130, -60)
        ax.set_title("Box already in place; tray going in (laser plate hidden so you can see)", fontsize=13, weight="bold")
    step_page(4, "tray in, then 4 long screws", [
        "Pull the PAN SERVO's plug up out of the middle of the box so it points up.",
        "Slide the TRAY in over the top edge of the box (there's about 4 cm under the laser plate), rails facing up.",
        "Thread the pan plug up through the tray's SQUARE HOLE, then let the tray drop onto the 4 corner posts.",
        "4x M3x20 down through the tray's corner holes and the posts into the turntable. Snug, not tight.",
        "The 2 back screws sit under the laser plate: use the short end of an L-shaped hex key, and tilt the plate by hand for room."],
        draw, "4_tray_and_screws.png")


# ------------------------------------------------------------------ 5. plug in
def step5():
    from matplotlib.patches import Rectangle

    def draw(ax):
        render(ax, trimesh.util.concatenate([m.copy() for m in (P.turntable(), P.parts["tower_servo"], P.parts["tower_bearing"],
                                                                 P.parts["ebox"], P.parts["ebox_tray"])]), (0, 0, -1), (0, 1, 0))
        ax.set_xlim(-118, 118); ax.set_ylim(-112, 112)

        def wire(pts, color, lw=3.4, ls="-"):
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=11)
            ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5), zorder=11)
        ax.add_patch(Rectangle((bxr[0], byr[0]), bxr[1] - bxr[0], 6, fc=HEX["board"], ec="#0d2a6b", zorder=9))
        ax.add_patch(Rectangle((-76, -41), 54, 82, fc="#fafafa", ec="#666", lw=1.5, zorder=8))
        ax.text(-49, 0, "BREAD-\nBOARD", ha="center", va="center", fontsize=10, weight="bold", zorder=9)
        ax.add_patch(Rectangle((-84, -33), 7, 8, fc="#d62728", ec="#7a0000", zorder=9))
        ax.text(-80, -44, "cable\nslot", ha="center", va="top", fontsize=9.5, weight="bold", color="#7a0000", zorder=9)
        ax.text(0, -P.A - 32, "TILT-SERVO TOWER", ha="center", fontsize=10.5, weight="bold", color="#1d4f91", zorder=12)
        ax.text(0, P.A + 30, "BEARING TOWER", ha="center", fontsize=10.5, weight="bold", color="#555", zorder=12)
        ax.text(47, byr[0] + 13, "SERVO BOARD", ha="center", fontsize=10, weight="bold", color=HEX["board"], zorder=12)
        ax.annotate("", xy=(114, 0), xytext=(86, 0), arrowprops=dict(arrowstyle="-|>", lw=3, color="#444"))
        ax.text(100, 7, "FRONT\n(laser)", ha="center", fontsize=9.5, color="#444")
        wire([(14.5, -2), (16, -18), (30, -30), (34, byr[0] + 6)], "#7b2cbf")
        wire([(22, -P.A - 34), (43, -P.A - 6), (43, byr[0] + 6)], "#1f77b4")
        wire([(60, byr[0] + 5), (-12, -32), (-21, -30), (-79, -30)], "#2ca02c")
        wire([(30, 4), (-10, 6), (-22, 6), (-36, 6)], "#d62728")
        ax.set_title("Seen from straight above, lid off, laser plate hidden", fontsize=13, weight="bold")
        for i, (t, c) in enumerate([("pan servo plug -> board", "#7b2cbf"), ("tilt servo plug -> board (in over the notch)", "#1f77b4"),
                                    ("12 V + USB -> out the back opening -> cable slot", "#2ca02c"), ("laser wires -> breadboard (with slack)", "#d62728")]):
            ax.plot([0.0, 0.05], [0.16 - i * 0.045] * 2, transform=ax.transAxes, color=c, lw=4, zorder=30)
            ax.text(0.06, 0.16 - i * 0.045, t, transform=ax.transAxes, fontsize=11, va="center", zorder=30)
    step_page(5, "plug everything back in", [
        "PAN plug (purple): from the tray's square hole into the SERVO BOARD.",
        "TILT plug (blue): from the TILT-SERVO TOWER, in over the box's notch, into the SERVO BOARD.",
        "12 V + USB (green): out the box's low back opening, down the turntable's cable slot like before.",
        "Laser wires (red): from the LASER PLATE over the back of the box to the BREADBOARD. Leave a loose loop so the plate can tilt."],
        draw, "5_plug_in.png", size=(13, 12))


# ------------------------------------------------------------------ 6. test + lid
def step6():
    lift = 30.0
    lid = moved(P.parts["ebox_lid"], (0, 0, lift))
    sc = [screw_mesh(h + Z * (lift + 25), a, L) for h, a, L in screws_named("lid -> box")]
    box = trimesh.util.concatenate([P.parts["ebox"], P.parts["ebox_tray"], BOARD])

    def draw(ax):
        proj = render_multi(ax, turret(plate=False) + [(box, COL["box"]), (lid, COL["lid"])] + [(m, COL["screw"]) for m in sc], d=D_VIEW)
        for x, y in P.EB_SCREWS:
            path(ax, proj, np.array((x, y, P.EB_Z1 + lift + 20)), np.array((x, y, P.EB_Z1 + 1)))
        tag(ax, proj, np.array((P.EB_X1, 0, P.EB_Z1 + lift)), "LID: slide in over the box,\ncut-out edge over the notch", 120, 40, color="#6b3fb0")
        tag(ax, proj, np.array((73.5, -37.5, P.EB_Z1 + lift + 25)), "4x M3x8 (short screws)", 120, -20)
        ax.set_title("Only after the test works (laser plate hidden so you can see)", fontsize=13, weight="bold")
    step_page(6, "test, then the lid", [
        "TEST FIRST: run turret_teleop.bat --port COM9 --laser COM11. Pan and tilt all the way to every limit. Watch that no wire pulls tight or rubs the box.",
        "Fix any tight wire now, while you can still reach inside.",
        "Slide the LID in over the box (same gap under the laser plate), lip down, the edge with the cut-out over the notch so the tilt wire isn't pinched.",
        "4x M3x8 into the inserts from step 1. The lid's 2 round holes are for the webcam and laser wires later; the 2 round holes in the box's front wall are for a future arm switch. Leave them all empty for now."], draw, "6_test_and_lid.png")


if __name__ == "__main__":
    parts(); step1(); step2(); step3(); step4(); step5(); step6()
