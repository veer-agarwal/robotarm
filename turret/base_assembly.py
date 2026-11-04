"""Exploded pictures: base_floor + base_frame, then the turntable onto the base (post into the pan horn)."""
import numpy as np
import trimesh
import perses_turret as P
from assembly_map import (crop, render_multi, tag, path, page, screw_mesh, screws_named,
                          C_PRINT, C_PART, C_SERVO, C_SCREW, C_BRG)

Z = np.array((0, 0, 1.0))
C_FLOOR = np.array([0.62, 0.78, 0.92])


def base():
    floor = P.base_floor()
    frame = trimesh.util.concatenate([P.base_frame(), P.cable_tray()])
    up = 60
    frame.apply_translation(Z * up)
    sc = [screw_mesh(h - Z * 30, a, L) for h, a, L in screws_named("base floor")]
    corners = [np.array((x, y, P.FLOOR_T)) for x, y in P.FRAME_SCREWS]

    def draw(ax):
        proj = render_multi(ax, [(floor, C_FLOOR), (frame, C_PRINT)] + [(m, C_SCREW) for m in sc], d=(0.5, 0.62, 0.6))
        for c in corners:
            path(ax, proj, c - Z * 30, c + Z * (up - 2))
        tag(ax, proj, corners[0] - Z * 36, "4x M3x6 from UNDER the floor", -60, -40)
        tag(ax, proj, np.array((0, 0, P.FLOOR_T)), "base_floor (seen from below)", 0, -95)
        tag(ax, proj, np.array((-100, 0, up + 25)), "base_frame + cable tray\n(cable opening on this side)", -90, 50)

    page("Base: base_floor under base_frame (view from BELOW)",
         ["Set base_frame deck-side up on top of base_floor. The post on base_floor ends up in the middle, right under the deck's big round hole.",
          "Line up the frame's cable opening (the gap in one side of the skirt) with the floor's matching side, so the floor's 4 small holes near the edges line up with the frame's 4 inserts.",
          "Tip it up and drive 4x M3x6 from under the floor into the frame's inserts. Snug.",
          "The floor's 4 holes near the corners (bigger, M4) are for screwing the turret to a desk later: optional."],
         draw, "6_base_assembly.png")


def turntable_on_base():
    floor = crop(P.base_floor(), ymax=0)
    frame = crop(trimesh.util.concatenate([P.base_frame(), P.cable_tray()]), ymax=0)
    up = 75
    top = [P.turntable(), P.parts["pan_floor_cap"], P.parts["tower_servo"], P.parts["tilt_floor_cap"]]
    top += [P.parts[f"roller_axle_{k}"] for k in (1, 2, 3)]
    tt = trimesh.util.concatenate([m.copy() for m in top])
    brg = trimesh.util.concatenate([P.hardware[f"bearing_roller_{k}"] for k in (1, 2, 3)])
    sv = trimesh.util.concatenate([P.hardware["servo_pan"], P.hardware["servo_tilt"]])
    for m in (tt, brg, sv):
        m.apply_translation(Z * up)
    hs = [screw_mesh(h - Z * 40, a, L) for h, a, L in screws_named("base post")]
    post_top = np.array((0, 0, P.PAN_ZR - P.HORN_FACE))      # horn face = post top

    def draw(ax):
        proj = render_multi(ax, [(floor, C_FLOOR), (frame, C_PRINT), (tt, C_PRINT), (brg, C_BRG), (sv, C_SERVO)] +
                            [(m, C_SCREW) for m in hs], d=(0.25, 0.85, -0.42))
        path(ax, proj, post_top + Z * (up - 3), post_top + Z * 2)
        tag(ax, proj, post_top + Z * (up - 2), "pan horn (bottom of the turntable)", 120, 15)
        tag(ax, proj, post_top, "post top (cut-away view)", 120, -10)
        tag(ax, proj, post_top - Z * 44, "4x M3x6 UP through the post\ninto the horn (long hex key)", -120, -25)
        tag(ax, proj, np.array((66, 0, P.ROLL_Z + up)), "3 rollers land on the deck,\ninside the cable tray", 120, 30)

    page("Turntable onto the base: pan horn sits on the post",
         ["Pan servo centered (2048) and in its sleeve, floor cap on, rollers on. Don't turn the horn from here on.",
          "Feed the turret's cables down through the turntable slot. Lower the turntable straight down: the horn lands on the post's flat top, the 3 rollers on the deck inside the cable tray.",
          "Rotate the turntable gently (not the horn) until the post's 4 holes line up with the horn's 4 threaded holes. Laser pointing forward = the cable opening side behind.",
          "From under the base, through the floor's openings: 4x M3x6 up through the post into the horn. Snug. The turntable should sit level on all 3 rollers."],
         draw, "7_turntable_on_base.png")


if __name__ == "__main__":
    base(); turntable_on_base()
