"""Exploded picture: cable tray onto the base_frame deck (3 pegs, notch over the cable hole)."""
import numpy as np
import perses_turret as P
from assembly_map import render_multi, tag, path, page, C_PRINT, C_PART


def cable_tray():
    z = np.array((0, 0, 1.0))
    frame = P.base_frame()
    tray = P.cable_tray()
    lift = 55
    tray.apply_translation(z * lift)
    pegs = [np.array((89 * np.cos(np.radians(a)), 89 * np.sin(np.radians(a)), P.DECK_TOP)) for a in (0, 120, 240)]
    notch = np.array((-89.0, -32.0, P.DECK_TOP))

    def draw(ax):
        proj = render_multi(ax, [(frame, C_PRINT), (tray, C_PART)], d=(0.45, 0.62, -0.64))
        for p in pegs:
            path(ax, proj, p + z * (lift - 4), p + z * 1)
        tag(ax, proj, pegs[0] + z * (lift - 3), "peg", 40, 20, color="#b5651d")
        tag(ax, proj, pegs[1] + z * (lift - 3), "peg", -30, 30, color="#b5651d")
        tag(ax, proj, pegs[2] + z * (lift - 3), "peg", 40, -25, color="#b5651d")
        tag(ax, proj, notch + z * lift, "notch in the tray", -70, 35, color="#b5651d")
        tag(ax, proj, notch, "cable hole in the deck\n(the notch goes right over it)", -80, -40)
        tag(ax, proj, np.array((-100, 0, 20)), "cable opening\n(-x side of the base)", -140, 10)

    page("Cable tray onto the base_frame deck",
         ["Base_frame deck side up (the flat top with the big round hole). The tray goes on upside-down from how it printed: walls UP, pegs DOWN.",
          "Turn the tray until its notch sits right over the deck's rectangular cable hole (same side as the base's cable opening). "
          "Only 1 of the 3 positions lines up.",
          "Push the 3 pegs into the 3 small holes. It should sit flat without rocking: if a peg is tight, twist it in or shave it.",
          "Lift it off, a small dab of glue (superglue or hot glue) on each peg, press it back in. Keep glue off the deck where the rollers run (inside the tray)."],
         draw, "5_cable_tray.png")


if __name__ == "__main__":
    cable_tray()
