"""Top-view layout: where the breadboard, servo board and every cable go on the turntable (temporary, before the ebox)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import perses_turret as P
from insert_map import render

OUT = os.path.join(P.HERE, "out", "assembly")
os.makedirs(OUT, exist_ok=True)


def box(ax, x0, y0, x1, y1, fc, ec, label=None, ls="-", alpha=0.9, lw=1.8, fs=9.5, tc="#111", z=6):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=lw, ls=ls, alpha=alpha, zorder=z))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center", fontsize=fs, color=tc, weight="bold", zorder=z + 1)


def wire(ax, pts, color, lw=3.2, label=None, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=9, label=label)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=lw * 0.8), zorder=9)


def main():
    fig = plt.figure(figsize=(13, 12.5), dpi=140)
    fig.suptitle("Turret wiring layout (top view, temporary until the ebox)", fontsize=19, weight="bold", y=0.975)
    ax = fig.add_axes([0.03, 0.31, 0.62, 0.62])
    render(ax, P.turntable(), (0, 0, -1), (0, 1, 0))
    ax.set_xlim(-118, 118); ax.set_ylim(-112, 112)

    A = P.A
    # towers (go on tomorrow)
    box(ax, -32, -A - 30, 32, -A, "#c9d3e3", "#55607a", "tower_servo foot", ls="--", alpha=0.55)
    box(ax, -32, A, 32, A + 30, "#c9d3e3", "#55607a", "tower_bearing foot", ls="--", alpha=0.55)
    # pan floor cap
    box(ax, P.SV_AX - P.SLV_HX, -P.SLV_HY, P.SV_AX + P.SLV_HX, P.SLV_HY, "#f2b46b", "#a8621b", "pan floor cap", alpha=0.85)
    # servo board: standing on tower_servo's inner face
    box(ax, -21, -A + 0.2, 21, -A + 4.0, "#2e7d32", "#123d14", None, alpha=1, lw=1.5, z=8)
    ax.annotate("SERVO BOARD\nstanding up on tower_servo's\ninner face, 40-70 mm above\nthe turntable, servo ports UP",
                xy=(0, -A + 2), xytext=(0, -98), ha="center", fontsize=9.5, weight="bold", color="#1b5e20",
                arrowprops=dict(arrowstyle="-", color="#1b5e20", lw=1.2), zorder=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1b5e20"))
    # breadboard + ESP32 behind the cap (-x)
    box(ax, -73, -41, -19, 41, "#fafafa", "#666", None, alpha=0.95, z=7)
    box(ax, -62, -14, -30, 14, "#222", "#000", "ESP32", alpha=1, tc="white", z=8)
    ax.text(-46, 30, "breadboard", ha="center", fontsize=10, weight="bold", zorder=9)
    ax.text(-46, -33, "(+ transistor, 1k)", ha="center", fontsize=8.5, zorder=9)
    # cable slot
    box(ax, -84, -33, -77, -25, "#d62728", "#7a0000", None, alpha=1, z=8)
    ax.annotate("cable slot", xy=(-80.5, -29), xytext=(-104, -66), fontsize=9.5, weight="bold", color="#7a0000",
                arrowprops=dict(arrowstyle="-", color="#7a0000"), zorder=12)
    # forward arrow
    ax.annotate("", xy=(112, 0), xytext=(84, 0), arrowprops=dict(arrowstyle="-|>", lw=3, color="#444"))
    ax.text(98, 8, "laser\npoints\nthis way", ha="center", fontsize=9, color="#444")

    # wires
    cap_exit = (P.SV_AX + 2, -6)
    wire(ax, [cap_exit, (14, -30), (8, -A + 6)], "#7b2cbf", label="pan servo lead (out of the cap's cutout)")
    wire(ax, [(22, -A - 34), (22, -A - 2), (14, -A + 6)], "#1f77b4", label="tilt servo lead (from the tower_servo cap, over the tower)")
    wire(ax, [(28, 4), (0, 6), (-22, 6)], "#d62728", label="laser red/black (down from the laser plate, with slack)")
    wire(ax, [(-18, -A + 3), (-52, -A + 3), (-76, -31)], "#2ca02c", label="servo board 12 V + USB")
    wire(ax, [(-60, -14), (-68, -24), (-78, -29)], "#555", label="ESP32 USB", ls="--")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10, frameon=False)

    notes = ("Under the turntable: the 3 cables (12 V, 2x USB) drop through the slot into the cable tray. Leave a loose loop\n"
             "most of the way around the tray, then out the base's cable opening. That loop lets the turret pan +-120 deg.\n"
             "If a plug won't fit through the slot, run that cable over the turntable's edge into the tray instead.\n\n"
             "Why the servo board stands on the tower: the tilt servo's stock lead is only 152 mm. Flat on the turntable it\n"
             "can't reach; at 40-70 mm up on tower_servo's inner face (where the ebox puts it) it does. Stay below ~110 mm\n"
             "up the tower so the laser plate's ear can't hit it when it tilts.\n\n"
             "Tonight (towers not on yet): the board can lie loose on the turntable for the pan wiggle test.")
    fig.text(0.05, 0.27, notes, fontsize=11, va="top", linespacing=1.5)
    fig.savefig(os.path.join(OUT, "8_wiring_layout.png"), facecolor="white")
    print("wrote 8_wiring_layout.png")


if __name__ == "__main__":
    main()
