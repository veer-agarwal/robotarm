"""Tello launch pad insert for the SO-101 hot-swap gripper socket.

Same frame as insert.stl / Moving_Jaw_SO101.stl: servo axis Z, jaw points -Y.
The pad plate hangs off the insert's plate bottom face (y = -44) and is offset toward -X,
away from the fixed jaw, so the open gripper clears it.
"""
import numpy as np
import trimesh
from trimesh.creation import box

PAD = 90.0          # pad side (mm); Tello is 98 x 92.5 with guards, ~80 x 80 body
T = 2.5             # pad thickness
RIM_H, RIM_W = 3.0, 2.0
X_OFF = -25.0       # pad centre offset toward -X (away from the fixed jaw)
Y_TOP = -44.0       # insert plate bottom face

def b(sx, sy, sz, cx, cy, cz):
    m = box((sx, sy, sz)); m.apply_translation((cx, cy, cz)); return m

insert = trimesh.load("../stl/gripper_socket/insert.stl")

y_pad = Y_TOP - T / 2
parts = [insert, b(PAD, T, PAD, X_OFF, y_pad, 0)]
# rim: 4 walls on the pad's outer face (the face the Tello sits on, y = -46.5)
y_rim = Y_TOP - T - RIM_H / 2
h = PAD / 2
parts += [
    b(PAD, RIM_H, RIM_W, X_OFF, y_rim, h - RIM_W / 2),
    b(PAD, RIM_H, RIM_W, X_OFF, y_rim, -h + RIM_W / 2),
    b(RIM_W, RIM_H, PAD, X_OFF + h - RIM_W / 2, y_rim, 0),
    b(RIM_W, RIM_H, PAD, X_OFF - h + RIM_W / 2, y_rim, 0),
]
# stiffening gusset between insert plate and pad (plate is 16 x 22, pad is offset)
parts.append(b(22, 3, 22, 0, Y_TOP - 1.5, 0))

pad = trimesh.boolean.union(parts, engine="manifold")
# weight-saving windows in the pad, clear of the insert footprint
cut = []
for cx in (X_OFF - 26, X_OFF + 26):
    for cz in (-26, 26):
        cut.append(b(26, 20, 26, cx, y_pad, cz))
pad = trimesh.boolean.difference([pad] + cut, engine="manifold")
pad.export("../stl/launch_pad/launch_pad_insert.stl")
print("bounds", pad.bounds.round(1), "volume cm3", round(pad.volume / 1000, 1),
      "approx grams PLA @40%", round(pad.volume / 1000 * 1.24 * 0.55, 1))
