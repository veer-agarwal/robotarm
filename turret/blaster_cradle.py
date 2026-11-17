"""Cradle for the Dart Zone Venom Pro, mounted UPSIDE DOWN on the tilt hinge (rail down, magazine up), with the
Arducam (your printed arducam_mount + keeper) and the laser on a bridge over the blaster's front section.

Three parts, all printable without supports:
  blaster_base    ears on the tilt axis + floor. Prints upside down (floor top on the bed, ears up).
  rail_clamp      the rail channel (rail_clamp_test slice 1 profile, 0.15 mm), 90 mm long, bolted under the floor
                  with 4x M3x10; two M3x14 lock screws come up from below through the floor into the clamp,
                  push the rail up against the 45 deg lips (no play) and sit between two teeth (no sliding).
                  Prints flat, channel up (like the test slice).
  cam_tower       post on the +y side (away from the kill switch) + bridge over the front section with the
                  Arducam mount inserts and the laser block. 2x M3x8 into floor inserts. Prints on its +y face.

Blaster position: balance point (105.8 mm from its back end) over the tilt axis. It slides in rail-first from
the back: the -y side (kill switch side) is open and the grip/magazine never pass under the bridge.

Measured blaster (mm, from its back end; heights from the rail top):
  length 226 | rail teeth 63.9-194.8 | rail 21.2 edge / 17.0 top / 15.8 neck, ~9 tall, 10.1 pitch
  body 52.6 wide | deep section to 136.9: 86.4 | slope to 160.8 | front section: 67.8 | muzzle centre 46.3
  grip 42-96 | loaded magazine tip 226.6 | kill switch at the front, -y side once inverted, 37 out

    python blaster_cradle.py    fit + tilt-sweep + slide-in checks, writes out/blaster_*.stl (print orientation)
"""
import os, sys
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix as Rot
from shapely.geometry import Polygon

sys.argv.append("--no-export")
import perses_turret as T
from perses_turret import box, cyl, U, D, W

# ---------------------------------------------------------------- blaster (turret frame: x forward, z up)
BAL = 105.8                         # balance point from the back end -> over the tilt axis (x = 0)
XB = -BAL                           # back end x
CLAMP_T = 6.0                       # clamp base under the rail
FLOOR_Z1 = T.CR_Z1                  # floor top
RAIL_Z = FLOOR_Z1 + CLAMP_T         # rail top face (teeth) rests here, blaster upside down above it
EDGE, NECK, RAIL_H = 21.2, 15.8, 9.0
BODY_HW = 52.6 / 2
X_DEEP1, X_FRONT0, X_FRONT1 = XB + 136.9, XB + 160.8, XB + 226.0
H_DEEP, H_FRONT, H_MUZZLE, H_MAG = 86.4, 67.8, 46.3, 226.6
GRIP_X = (XB + 42.0, XB + 96.2)
MUZZLE_Z = RAIL_Z + H_MUZZLE

# ---------------------------------------------------------------- cradle dimensions
FLOOR_X = (-45.0, 98.0)
CLAMP_X = (-42.0, 48.0)             # under the slotted rail (63.9..194.8 from back = -41.9..+89)
CLAMP_HW = 18.0
C, Z_W = 0.15, 3.5                  # slice 1 fit
LOCK_X = (-30.0, 30.0)              # lock screws (outside the ears, x +-20), in a tooth row
LOCK_Y = 6.0
CLAMP_BOLTS = [(x, sy * 14.0) for x in (CLAMP_X[0] + 6, CLAMP_X[1] - 6) for sy in (-1, 1)]
TW_X = (57.0, 97.0)                 # tower / bridge along x (bridge starts past the slope)
POST_Y = (BODY_HW + 3.7, 46.0)      # post on +y, 3.7 mm clear of the body
BR_Z0 = RAIL_Z + H_FRONT + 3.0      # bridge underside: 3 mm over the front section
BR_T = 8.0
BR_Z1 = BR_Z0 + BR_T
BR_Y0 = -30.0                       # bridge reaches over to -y (camera centred on the barrel)
FOOT_T = 6.0
TOWER_BOLTS = [(TW_X[0] + 7, 38.0), (TW_X[1] - 7, 38.0)]
CAM_X0 = 57.0                       # arducam_mount pad start (its pad is 40 long, screws 6 and 29 in)
CAM_INS = [(CAM_X0 + 6, 0.0), (CAM_X0 + 29, 0.0)]
LASER_X = (TW_X[0] + 2, TW_X[1] - 2)
LASER_Y = 38.0
LASER_Z = BR_Z1 + 11.0


def ins(x, y, z_top, depth=T.INSERT_DEPTH, up=True):
    """Heat-set insert hole opening at z_top (pointing up) or at z_top from below (up=False)."""
    return cyl(T.INSERT_D / 2, (x, y, z_top - depth if up else z_top - 1), (x, y, z_top + 1 if up else z_top + depth), 24)


def clr(x, y, z0, z1, r=T.M3_CLEAR / 2):
    return cyl(r, (x, y, z0), (x, y, z1), 24)


def extrude_x(poly, x0, x1):
    m = trimesh.creation.extrude_polygon(poly, x1 - x0)                      # (y, z, x)
    m.apply_transform(np.array([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float))
    return m


# ---------------------------------------------------------------- parts
def base():
    floor = box((FLOOR_X[0], -T.YO, T.CR_Z0), (FLOOR_X[1], T.YO, FLOOR_Z1))
    ears, cuts = T.ears_and_cuts()
    b = U(floor, *ears)
    for x, y in CLAMP_BOLTS:
        cuts.append(clr(x, y, T.CR_Z0 - 1, FLOOR_Z1 + 1))
        cuts.append(cyl(3.0, (x, y, T.CR_Z0 - 1), (x, y, T.CR_Z0 + 1.5), 24))     # head seat
    for x in LOCK_X:
        cuts.append(clr(x, LOCK_Y, T.CR_Z0 - 1, FLOOR_Z1 + 1))
    for x, y in TOWER_BOLTS:
        cuts.append(ins(x, y, FLOOR_Z1))
    cuts.append(box((-12, -32, T.CR_Z0 - 1), (12, -16, FLOOR_Z1 + 1)))            # cable pass (MR30 leads, -y)
    cuts.append(box((52, -T.YO - 1, T.CR_Z0 - 1), (FLOOR_X[1] + 1, -T.YO + 14, FLOOR_Z1 + 1)))   # kill switch side relief
    return D(b, *cuts)


def clamp_profile():
    a, n = EDGE / 2 + C, NECK / 2 + C
    z0 = RAIL_Z
    z_n = Z_W + (a - n)
    outer = Polygon([(-CLAMP_HW, FLOOR_Z1), (CLAMP_HW, FLOOR_Z1), (CLAMP_HW, z0 + 8.3), (-CLAMP_HW, z0 + 8.3)])
    chan = Polygon([(-a, z0), (a, z0), (a, z0 + Z_W), (n, z0 + z_n), (n, z0 + 10), (-n, z0 + 10), (-n, z0 + z_n), (-a, z0 + Z_W)])
    return outer.difference(chan)


def rail_clamp():
    c = extrude_x(clamp_profile(), *CLAMP_X)
    cuts = [ins(x, y, FLOOR_Z1, up=False) for x, y in CLAMP_BOLTS]
    cuts += [ins(x, LOCK_Y, FLOOR_Z1, depth=CLAMP_T + 1, up=False) for x in LOCK_X]     # lock screws come through
    for xe in CLAMP_X:                                                                   # lead-in chamfer at both ends
        s = 1 if xe == CLAMP_X[0] else -1
        cuts.append(W(box((-3, -EDGE / 2 - C - 1.2, RAIL_Z), (3, EDGE / 2 + C + 1.2, RAIL_Z + 8.4)),
                      Rot(0, (0, 0, 1), (0, 0, 0)) @ trimesh.transformations.translation_matrix((xe, 0, 0))))
    return D(c, *cuts)


def cam_tower():
    post = box((TW_X[0], POST_Y[0], FLOOR_Z1), (TW_X[1], POST_Y[1], BR_Z0 + 0.01))
    bridge = box((TW_X[0], BR_Y0, BR_Z0), (TW_X[1], POST_Y[1], BR_Z1))
    block = box((LASER_X[0], LASER_Y - 8, BR_Z1 - 0.01), (LASER_X[1], LASER_Y + 8, BR_Z1 + 22))
    t = U(post, bridge, block)
    cuts = [box((TW_X[0] + 7, POST_Y[0] - 1, FLOOR_Z1 + FOOT_T + 6), (TW_X[1] - 7, POST_Y[1] + 1, BR_Z0 - 22))]   # lighten
    for x, y in TOWER_BOLTS:
        cuts.append(clr(x, y, FLOOR_Z1 - 1, FLOOR_Z1 + FOOT_T + 1))
        cuts.append(box((x - 3.5, y - 4, FLOOR_Z1 + FOOT_T), (x + 3.5, POST_Y[1] + 1, FLOOR_Z1 + FOOT_T + 6)))   # head pocket, open +y
    for x, y in CAM_INS:
        cuts.append(ins(x, y, BR_Z1))
    cuts.append(cyl(10.5 / 2 + 0.15, (LASER_X[0] - 1, LASER_Y, LASER_Z), (LASER_X[1] + 1, LASER_Y, LASER_Z)))    # laser bore
    cuts.append(ins((LASER_X[0] + LASER_X[1]) / 2, LASER_Y, BR_Z1 + 22))                                         # set screw
    cuts.append(box((LASER_X[0] - 1, LASER_Y - 1.5, BR_Z1 - 7), (LASER_X[0] + 6, LASER_Y + 1.5, LASER_Z)))        # lead exit
    cuts.append(box((LASER_X[0] - 1, LASER_Y - 1.5, FLOOR_Z1 + FOOT_T + 6), (LASER_X[0] + 3, LASER_Y + 1.5, BR_Z1 - 6)))  # lead channel down the post
    return D(t, *cuts)


# ---------------------------------------------------------------- blaster proxy (for checks)
def blaster(dx=0.0):
    rail_poly = Polygon([(-8.5, 0), (8.5, 0), (EDGE / 2, 2.1), (EDGE / 2, Z_W - 0.1), (NECK / 2, Z_W - 0.1 + 2.7),
                         (NECK / 2, RAIL_H), (-NECK / 2, RAIL_H), (-NECK / 2, Z_W - 0.1 + 2.7), (-EDGE / 2, Z_W - 0.1), (-EDGE / 2, 2.1)])
    rp = Polygon([(y, z + RAIL_Z) for y, z in rail_poly.exterior.coords])
    parts = [extrude_x(rp, XB, X_FRONT1),
             box((XB, -BODY_HW, RAIL_Z + RAIL_H), (X_DEEP1, BODY_HW, RAIL_Z + H_DEEP)),
             trimesh.convex.convex_hull(np.array([[x, y, z] for y in (-BODY_HW, BODY_HW) for x, z in
                                                  ((X_DEEP1, RAIL_Z + RAIL_H), (X_DEEP1, RAIL_Z + H_DEEP), (X_FRONT0, RAIL_Z + RAIL_H),
                                                   (X_FRONT0, RAIL_Z + H_FRONT))])),
             box((X_FRONT0, -BODY_HW, RAIL_Z + RAIL_H), (X_FRONT1, BODY_HW, RAIL_Z + H_FRONT)),
             box((GRIP_X[0], -16, RAIL_Z + H_DEEP - 1), (GRIP_X[1], 16, RAIL_Z + H_MAG)),
             box((X_FRONT1 - 26, -BODY_HW - 37, RAIL_Z + 2), (X_FRONT1 - 6, -BODY_HW + 0.5, RAIL_Z + 24))]   # kill switch
    m = U(*parts)
    m.apply_translation((dx, 0, 0))
    return m


def vol(a, b):
    if not (np.all(a.bounds[0] <= b.bounds[1]) and np.all(b.bounds[0] <= a.bounds[1])): return 0.0
    r = trimesh.boolean.intersection([a, b], engine="manifold")
    return 0.0 if r.is_empty else r.volume


if __name__ == "__main__":
    import arducam_mount as AM
    shift = (CAM_X0 - (AM.RX0 + 2), 0, BR_Z1 - AM.ZR)                       # arducam_mount riser frame -> bridge top
    cam_parts = []
    for p in (AM.mount(), AM.keeper(), AM.camera_proxy()):
        q = p.copy(); q.apply_translation(shift); cam_parts.append(q)
    cam = U(*cam_parts)
    laser = cyl(10.5 / 2, (LASER_X[0] + 1, LASER_Y, LASER_Z), (LASER_X[1] + 1, LASER_Y, LASER_Z))
    bse, clp, twr, bl = base(), rail_clamp(), cam_tower(), blaster()
    for n, m in (("base", bse), ("clamp", clp), ("tower", twr)):
        print(f"{n}: watertight={m.is_watertight}  vol {m.volume / 1000:.1f} cm3  extents {np.round(m.extents, 1)}")
    pairs = [("blaster x base", bl, bse), ("blaster x clamp", bl, clp), ("blaster x tower", bl, twr),
             ("blaster x camera", bl, cam), ("clamp x base", clp, bse), ("tower x base", twr, bse),
             ("tower x clamp", twr, clp), ("camera x tower", cam, twr), ("laser x tower", laser, twr)]
    for n, a, b in pairs:
        print(f"  {n:18s} {vol(a, b):8.2f} mm3")
    # slide-in path from the back (blaster comes from -x)
    worst = max(vol(blaster(-dx), twr) + vol(blaster(-dx), cam) for dx in range(0, 200, 10))
    print(f"slide-in from the back, 0..200 mm: worst overlap with tower/camera {worst:.2f} mm3")
    # tilt sweep against everything that doesn't tilt
    fixed = {k: v for k, v in {**T.parts, **T.hardware}.items() if k not in T.TILTING and k != "laser_plate"}
    moving = {"base": bse, "clamp": clp, "tower": twr, "camera": cam, "blaster": bl}
    bad = []
    for deg in range(-20, 21, 5):
        M = Rot(np.radians(deg), (0, 1, 0), point=(0, 0, T.ZT))
        for nm, part in moving.items():
            p = part.copy(); p.apply_transform(M)
            for k, v in fixed.items():
                x = vol(p, v)
                if x > 1.0: bad.append((deg, nm, k, round(x, 1)))
    print("tilt sweep +-20:", "nothing touches" if not bad else bad)
    print(f"muzzle centre {MUZZLE_Z - T.ZT:.1f} mm above the tilt axis; lens {AM.BC_Z + shift[2] - MUZZLE_Z:.1f} mm above the muzzle line;"
          f" laser {LASER_Z - MUZZLE_Z:.1f} mm above, {LASER_Y:.0f} mm to +y")
    os.makedirs(T.OUT, exist_ok=True)
    b2 = bse.copy(); b2.apply_transform(Rot(np.pi, (1, 0, 0))); b2.apply_translation(-b2.bounds[0])          # floor top on the bed
    c2 = clp.copy(); c2.apply_translation(-c2.bounds[0])                                                   # flat, channel up
    t2 = twr.copy(); t2.apply_transform(Rot(-np.pi / 2, (1, 0, 0))); t2.apply_translation(-t2.bounds[0])     # +y face on the bed
    for n, m in (("blaster_base", b2), ("blaster_rail_clamp", c2), ("blaster_cam_tower", t2)):
        m.export(os.path.join(T.OUT, n + ".stl"))
    U(bse, clp, twr, cam, laser, bl).export(os.path.join(T.OUT, "blaster_preview.stl"))
    print("wrote", T.OUT)
