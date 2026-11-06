"""Arducam OV9281 (B0592, UC-844 board, 38 x 38 mm) mount for the laser plate's webcam riser.

Sits on the riser top, held by two zip ties (or the velcro strap) through the riser's 20 mm slots. The board
stands on 4 posts in front of a U-frame, lens forward, 8 mm off the frame so the parts and the 4-pin USB plug
on its back clear. Each post has pilot holes for BOTH hole patterns the board takes (34 x 34 and 28 x 28, per
Arducam's datasheet), so whichever holes your board has, one set lines up. M2 x 6-8 screws, self-tapping into
the 1.8 mm pilots.

    python arducam_mount.py           writes out/arducam_mount.stl (print orientation) + a tilt-sweep fit check

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
PATTERNS = (34.0, 28.0)
PILOT = 1.8
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
    posts, pilots = [], []
    for sy in (-1, 1):
        for sz in (-1, 1):
            a = [(sy * p / 2, sz * p / 2) for p in PATTERNS]
            pts = []
            for (y, z) in a:
                c = cyl(2.7, (FX1 - 0.5, y, BC_Z + z), (BX0, y, BC_Z + z), 32)
                pts.append(c)
            posts.append(trimesh.convex.convex_hull(np.vstack([p.vertices for p in pts])))
            for (y, z) in a:
                pilots.append(cyl(PILOT / 2, (BX0 - 7.0, y, BC_Z + z), (BX0 + 1, y, BC_Z + z), 24))
    m = U(pad, D(frame, window), *gussets, *posts)
    slots = [box((-46.0, s * 22 - 2.0, ZR - 1), (-32.0, s * 22 + 2.0, ZR + PAD_T + 1)) for s in (-1, 1)]   # zip ties
    return D(m, *pilots, *slots)


def camera_proxy():
    """Board, lens (M12 fisheye ~16 mm across, ~22 mm long) and the plug + cable bend on the back."""
    b = box((BX0, -BOARD / 2, BC_Z - BOARD / 2), (BX0 + BOARD_T, BOARD / 2, BC_Z + BOARD / 2))
    lens = cyl(8.0, (BX0 + BOARD_T, 0, BC_Z), (BX0 + BOARD_T + 22, 0, BC_Z))
    plug = box((FX0 - 6, -10.5, BC_Z - BOARD / 2 + 1), (BX0, -2.0, BC_Z - BOARD / 2 + 10))
    return U(b, lens, plug)


if __name__ == "__main__":
    m, cam = mount(), camera_proxy()
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
        for nm, part in (("mount", m), ("camera", cam)):
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
    U(plate, m, cam).export(os.path.join(T.OUT, "arducam_on_plate_preview.stl"))
    print("wrote", os.path.join(T.OUT, "arducam_mount.stl"))
