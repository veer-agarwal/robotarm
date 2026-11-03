"""Exploded picture: pan servo into the turntable sleeve, floor cap on top."""
import numpy as np
import perses_turret as P
from assembly_map import (crop, render_multi, tag, path, page, screw_mesh, screws_named,
                          C_PRINT, C_PART, C_SERVO, C_SCREW)


def pan_servo():
    z = np.array((0, 0, 1.0))
    tt = crop(P.turntable(), xmin=-95, xmax=80, ymin=-60, ymax=60)
    servo = P.hardware["servo_pan"].copy()
    cap = P.parts["pan_floor_cap"].copy()
    sc = [screw_mesh(h, a, L) for h, a, L in screws_named("pan_floor_cap")]
    E = {"servo": 70, "cap": 125, "sc": 160}
    servo.apply_translation(z * E["servo"]); cap.apply_translation(z * E["cap"])
    for m in sc: m.apply_translation(z * E["sc"])
    c0 = np.array((P.SV_AX, 0, P.TT_Z1))          # servo body centre on the top face

    def draw(ax):
        proj = render_multi(ax, [(tt, P_COL), (servo, C_SERVO), (cap, C_PART)] + [(m, C_SCREW) for m in sc],
                            d=(-0.55, 0.65, -0.52))
        path(ax, proj, c0 + z * 60, c0 + z * 4)
        tag(ax, proj, c0 + z * (E["sc"] + 8), "4x M3x12", 90, 10)
        tag(ax, proj, c0 + z * (E["cap"] + 4) + np.array((25, 0, 0)), "pan floor cap\n(cable cutout over the lead)", 110, -10, color="#b5651d")
        tag(ax, proj, c0 + z * (E["servo"] - 10) + np.array((-20, 0, 0)), "pan servo\nHORN DOWN", -110, 0)
        tag(ax, proj, np.array((0, 0, 46)) + z * E["servo"], "horn is on the BOTTOM\n(round window side)", 120, -55)
        tag(ax, proj, c0 + np.array((-24, 18, 0)), "turntable sleeve\n(inserts 1-4 on its rim)", -60, 75)
        tag(ax, proj, np.array((-80.5, -29, P.TT_Z1)), "cable slot", -50, -30)

    page("Pan servo: drops into the turntable from the TOP, horn down",
         ["Center it first: turret_setup.bat center --port COM9 --id 7, press Enter, unplug. Don't turn the horn after this.",
          "Turntable top face up. Lower the servo into the sleeve HORN FIRST: the horn comes out the bottom window (it later sits on the base's post).",
          "If the horn hits solid plastic instead of the round window, spin the servo 180 deg. Push it down until the body sits on the bottom plate.",
          "Lead: out the top of the sleeve, over to the cable slot. Cap on top with its cutout over the lead, 4x M3x12 into inserts 1-4, snug in an X pattern."],
         draw, "4_pan_servo.png")


P_COL = C_PRINT
if __name__ == "__main__":
    pan_servo()
