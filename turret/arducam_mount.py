"""Arducam OV9281 (B0592, UC-844 board, 38 x 38 mm) mount for the laser plate's webcam riser.

Sits on the riser top, held by 2x M3x8 screws into heat-set inserts in the riser top (CAM_INSERT_X in
perses_turret.py). The slots for zip ties are still there as a spare.

No M2 screws needed for the board: it slides down into two grooved rails (1.5 mm of each side edge held), lens
forward, 8 mm in front of a U-frame so the parts and the 4-pin USB plug on its back clear. A keeper bar sits on
the rail tops with two tongues that press on the board's top edge, held by 2x M3x8 into inserts in the rails.

    python arducam_mount.py    writes out/arducam_mount.stl + out/arducam_keeper.stl (print orientation) + fit check

World frame as perses_turret.py: x forward, z up, tilt axis along y at z = ZT.
"""
import os, sys
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix as Rot

sys.argv.append("--no-export")
import perses_turret as T
from perses_turret import box, cyl, U, D

ZR = T.CR_Z1 + T.CAM_RISER            # riser top (254)
RX0, RX1 = -T.LP_X, -18.0              # riser x span
PAD_T = 4.0
GAP = 8.0                              # board back to frame front (USB plug + parts on the back)
FRAME_T = 3.0
BOARD, BOARD_T = 38.0, 1.6
BC_Z = ZR + PAD_T + 1.0 + BOARD / 2    # board centre height (board bottom 1 mm above the pad)
FX1 = RX1                              # frame front face, flush with the riser front
FX0 = FX1 - FRAME_T
BX0 = FX1 + GAP                        # board back face
GRIP = 1.5                             # how much of each side edge the rails hold
RAIL_IN, RAIL_OUT = BOARD / 2 - GRIP, 24.5
LIP_X = BX0 + BOARD_T + 1.6            # front of the rails' lips
RAIL_TOP = BC_Z + BOARD / 2 + 3.0
KEEP_T = 4.0
KEEP_SCREW = (-14.0, 20.75)            # (x, |y|) of the keeper screws / rail inserts
WIN = 11.0                             # half-width of the frame's window (plug + cable pass through)


def mount():
    pad = box((RX0 + 2, -28, ZR), (RX1, 28, ZR + PAD_T))
    frame = box((FX0, -22, ZR), (FX1, 22, BC_Z + 22))
    window = box((FX0 - 1, -WIN, ZR + PAD_T), (FX1 + 1, WIN, BC_Z + WIN))
    gussets = []
    for y in (-20.5, 17.5):
        pts = np.array([[FX0, y, ZR + PAD_T], [FX0 - 10, y, ZR + PAD_T], [FX0, y, BC_Z + 12],
                        [FX0, y + 3, ZR + PAD_T], [FX0 - 10, y + 3, ZR + PAD_T], [FX0, y + 3, BC_Z + 12]])
        gussets.append(trimesh.convex.convex_hull(pts))
    rails, pilots = [], []
    for sy in (-1, 1):
        y0, y1 = sorted((sy * RAIL_IN, sy * RAIL_OUT))
        rails.append(box((FX1 - 0.5, y0, ZR), (LIP_X, y1, RAIL_TOP)))
        g0, g1 = sorted((sy * (RAIL_IN - 1.0), sy * (BOARD / 2 + 0.4)))
        pilots.append(box((BX0 - 0.15, g0, ZR + PAD_T + 1.0), (BX0 + BOARD_T + 0.15, g1, RAIL_TOP + 1)))   # groove
        x, y = KEEP_SCREW
        pilots.append(cyl(T.INSERT_D / 2, (x, sy * y, RAIL_TOP - T.INSERT_DEPTH), (x, sy * y, RAIL_TOP + 1), 24))
    m = U(pad, D(frame, window), *gussets, *rails)
    slots = [box((-46.0, s * 22 - 2.0, ZR - 1), (-32.0, s * 22 + 2.0, ZR + PAD_T + 1)) for s in (-1, 1)]   # zip ties (spare)
    screws = [cyl(T.M3_CLEAR / 2, (x, 0, ZR - 1), (x, 0, ZR + PAD_T + 1), 24) for x in T.CAM_INSERT_X]       # 2x M3x8 into the riser inserts
    return D(m, *pilots, *slots, *screws)


def keeper():
    """Bar across the rail tops; tongues drop into the grooves onto the board's top edge."""
    bar = box((FX1 - 0.5, -RAIL_OUT, RAIL_TOP), (LIP_X, RAIL_OUT, RAIL_TOP + KEEP_T))
    tongues = []
    for sy in (-1, 1):
        y0, y1 = sorted((sy * (RAIL_IN + 0.2), sy * (BOARD / 2 + 0.2)))
        tongues.append(box((BX0 + 0.1, y0, BC_Z + BOARD / 2 + 0.2), (BX0 + BOARD_T - 0.1, y1, RAIL_TOP + 0.01)))
    holes = [cyl(T.M3_CLEAR / 2, (KEEP_SCREW[0], sy * KEEP_SCREW[1], RAIL_TOP - 1), (KEEP_SCREW[0], sy * KEEP_SCREW[1], RAIL_TOP + KEEP_T + 1), 24)
             for sy in (-1, 1)]
    return D(U(bar, *tongues), *holes)


def camera_proxy():
    """Board, lens (M12 fisheye ~16 mm across, ~22 mm long) and the plug + cable bend on the back."""
    b = box((BX0, -BOARD / 2, BC_Z - BOARD / 2), (BX0 + BOARD_T, BOARD / 2, BC_Z + BOARD / 2))
    lens = cyl(8.0, (BX0 + BOARD_T, 0, BC_Z), (BX0 + BOARD_T + 22, 0, BC_Z))
    plug = box((FX0 - 6, -10.5, BC_Z - BOARD / 2 + 1), (BX0, -2.0, BC_Z - BOARD / 2 + 10))
    return U(b, lens, plug)


if __name__ == "__main__":
    m, cam, kp = mount(), camera_proxy(), keeper()
    print(f"keeper: watertight={kp.is_watertight}  extents={np.round(kp.extents, 1)}  keeper x mount: ", end="")
    _k = trimesh.boolean.intersection([kp, m], engine="manifold"); print(0.0 if _k.is_empty else round(_k.volume, 2), "mm3")
    print(f"mount: watertight={m.is_watertight}  vol={m.volume / 1000:.1f} cm3  extents={np.round(m.extents, 1)}")
    plate = T.parts["laser_plate"]

    def vol(a, b):
        if not (np.all(a.bounds[0] <= b.bounds[1]) and np.all(b.bounds[0] <= a.bounds[1])): return 0.0
        r = trimesh.boolean.intersection([a, b], engine="manifold")
        return 0.0 if r.is_empty else r.volume

    print("mount x laser plate:", round(vol(m, plate), 2), "mm3   camera x mount:", round(vol(cam, m), 2),
          "  camera x plate:", round(vol(cam, plate), 2), "  camera x laser:", round(vol(cam, T.hardware["laser"]), 2))
    fixed = {k: v for k, v in {**T.parts, **T.hardware}.items() if k not in T.TILTING}
    bad = []
    for deg in range(-int(T.TILT_LIMIT), int(T.TILT_LIMIT) + 1, 5):
        M = Rot(np.radians(deg), (0, 1, 0), point=(0, 0, T.ZT))
        for nm, part in (("mount", m), ("camera", cam), ("keeper", kp)):
            p = part.copy(); p.apply_transform(M)
            for k, v in fixed.items():
                x = vol(p, v)
                if x > 1.0: bad.append((deg, nm, k, round(x, 1)))
    print("tilt sweep +-%d deg:" % T.TILT_LIMIT, "OK, nothing touches" if not bad else bad)
    lens_z, laser_z = BC_Z, T.LASER_Z
    print(f"lens axis z {lens_z:.1f}, laser z {laser_z:.1f}: {lens_z - laser_z:.0f} mm above the beam")
    out = m.copy(); out.apply_translation(-out.bounds[0])          # pad on the bed as modelled
    os.makedirs(T.OUT, exist_ok=True)
    out.export(os.path.join(T.OUT, "arducam_mount.stl"))
    k2 = kp.copy(); k2.apply_transform(Rot(np.pi, (1, 0, 0))); k2.apply_translation(-k2.bounds[0])   # bar flat, tongues up
    k2.export(os.path.join(T.OUT, "arducam_keeper.stl"))
    U(plate, m, cam, kp).export(os.path.join(T.OUT, "arducam_on_plate_preview.stl"))
    print("wrote", os.path.join(T.OUT, "arducam_mount.stl"))
