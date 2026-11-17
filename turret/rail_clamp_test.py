"""Fit test for the Venom Pro rail clamp: three 25 mm slices of the future cradle's rail channel.

Measured rail (Venom Pro, toy Picatinny): 21.2 wide at the edges, 17.0 across the teeth, neck 15.8 where it
meets the body, ~9 tall, 45 deg undercut, 10.1 mm tooth pitch. The rail goes INTO the channel upside down
(teeth on the channel floor), the way the blaster will sit on the turret.

Two things aren't known exactly, so the three slices differ in both:
    slice   side clearance   undercut starts      marks on the side
      1        0.15 mm        3.5 mm from floor          1 dot   (snug)
      2        0.25 mm        3.8 mm                     2 dots
      3        0.40 mm        4.2 mm                     3 dots  (loose)

Print flat (channel opening up), no supports: the lips are 45 deg overhangs. PLA, 0.2 mm layers, 4 walls.
Slide each onto the rail from the FRONT end of the blaster. Pick the tightest one that slides on by hand and
doesn't rattle or lift off the rail.

    python rail_clamp_test.py      writes out/rail_clamp_test.stl (all three) + rail_clamp_test_<n>.stl
"""
import os
import numpy as np
import trimesh
from shapely.geometry import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

LEN = 25.0                     # along the rail
HALF_W = 18.0                  # outer half-width of the block
BASE = 4.0                     # floor thickness under the rail
TOP = 8.3                      # block top above the channel floor (rail is ~9 tall: stays clear of the body)
EDGE, NECK = 21.2, 15.8        # rail widest edge / neck widths
VARIANTS = [(0.15, 3.5), (0.25, 3.8), (0.40, 4.2)]   # (side clearance, height where the undercut starts)


def profile(c, z_w):
    """Block cross-section (y lateral, z up, channel floor at z = 0) with the rail channel cut out."""
    a, n = EDGE / 2 + c, NECK / 2 + c
    z_n = z_w + (a - n)                        # 45 deg lip down to the neck
    outer = Polygon([(-HALF_W, -BASE), (HALF_W, -BASE), (HALF_W, TOP), (-HALF_W, TOP)])
    chan = Polygon([(-a, 0), (a, 0), (a, z_w), (n, z_n), (n, TOP + 1), (-n, TOP + 1), (-n, z_n), (-a, z_w)])
    return outer.difference(chan)


def slice_part(k):
    c, z_w = VARIANTS[k]
    m = trimesh.creation.extrude_polygon(profile(c, z_w), LEN)       # extrudes along +z: (y, z_prof, x_len)
    m.apply_transform(np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float))  # -> x = length
    dots = [trimesh.creation.cylinder(radius=1.2, height=1.6, sections=24) for _ in range(k + 1)]
    for i, d in enumerate(dots):                                     # dimples on the +y side face
        d.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
        d.apply_translation((LEN / 2 + (i - k / 2) * 5.0, HALF_W, 1.0))
    m = trimesh.boolean.difference([m] + dots, engine="manifold")
    m.apply_translation(-m.bounds[0])
    return m


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    parts = []
    for k in range(3):
        p = slice_part(k)
        assert p.is_watertight
        p.export(os.path.join(OUT, f"rail_clamp_test_{k + 1}.stl"))
        q = p.copy(); q.apply_translation((0, k * (2 * HALF_W + 8), 0)); parts.append(q)
        c, z_w = VARIANTS[k]
        print(f"slice {k + 1}: clearance {c} mm, undercut from {z_w} mm, {p.extents.round(1)} mm, {p.volume / 1000:.1f} cm3")
    trimesh.util.concatenate(parts).export(os.path.join(OUT, "rail_clamp_test.stl"))
    print("wrote", os.path.join(OUT, "rail_clamp_test.stl"))
