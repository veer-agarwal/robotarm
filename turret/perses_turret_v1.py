"""
Pan-tilt turret for a Nerf Rival Perses (upright), driven by two Feetech STS3215 (12 V) servos.

Units: mm. World frame: z up, x forward (muzzle at pan = 0), y left. Pan axis = world z axis.

Hardware (existing order + M3 kit):
  - 2x STS3215 12 V (geometry from TheRobotStudio's sts3215_03a_v1.stl)
  - 4x 6802-2RS (15 x 24 x 5): 3 turntable rollers + 1 tilt idler
  - M3 socket head screws, M3 brass heat-set inserts (hole d4.0 x 6 deep, insert 5 long)

How each servo is held: it slides into a sleeve (7 mm walls, 0.25 mm clearance) whose closed end is a
4 mm plate with a d27 window for the horn. A floor cap screwed into heat-set inserts in the sleeve's
open end clamps the servo against that plate on two 2 mm ledges, clear of the bottom pad and connectors.
The servo case screws are not used (the front pair sit 13.2 mm from the output, under the horn).

python perses_turret.py      -> STLs in ./out
"""
import os, sys
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix as Rot, translation_matrix as Tr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SERVO_STL = os.environ.get("STS3215_STL", "/home/claude/SO-ARM100/Simulation/SO101/assets/sts3215_03a_v1.stl")

# ------------------------------------------------------------------ parameters
C = 0.25                    # clearance per side on fitted pockets
M3_CLEAR, M3_HEAD_D, M3_HEAD_H = 3.4, 5.5, 3.0
CB_D = M3_HEAD_D + 0.6      # counterbore
INSERT_D, INSERT_DEPTH, INSERT_LEN = 4.0, 6.0, 5.0
BRG_OD, BRG_ID, BRG_W = 24.0, 15.0, 5.0
BRG_SEAT = BRG_OD + 0.15
SHAFT_D = BRG_ID - 0.1

# STS3215, servo-local frame (official model): body x +-22.7, y +-12.4, faces z +-15.9; output axis at
# x = 12.5; horn face z = 18.6 with 4x M3 at (+-4.95, +-4.95) from the axis; centre screw head to z 20.2.
SV_X, SV_Y, SV_FACE, SV_AX = 22.7, 12.4, 15.9, 12.5
HORN_FACE, HP = 18.7, 4.95       # horn top measured at 18.7; threads 18.7 -> 16.1, body at 14.4
HORN_CLAMP = 3.0                  # M3x6: 3.0 clamp + 2.6 thread, tip 1.3 above the servo body
WIN_R = 13.5                # window in the plate over the servo
PLATE_T = 4.0               # plate over the servo face
HUB_R = 13.0
SLV_WALL = 7.0
SLV_HX, SLV_HY = SV_X + C + SLV_WALL, SV_Y + C + SLV_WALL            # 29.95 x 19.65 half sizes
INS_XY = [(sx * (SV_X + C + SLV_WALL / 2), sy * (SV_Y + C + SLV_WALL / 2)) for sx in (-1, 1) for sy in (-1, 1)]
FCAP_T = 8.0                # floor cap, servo-local z -15.9 .. -23.9

# Perses proxy (no published blaster dimensions: built from the 25 x 6 x 13 in box and photos)
P_L, P_W, P_H, P_MASS = 560.0, 75.0, 115.0, 1.6

# layout
DECK_Z0 = 45.0              # base deck underside = pan servo face
DECK_T = PLATE_T
DECK_TOP = DECK_Z0 + DECK_T                 # 49
PAN_HORN_Z = DECK_Z0 + (HORN_FACE - SV_FACE)  # 47.7
ROLL_Z = DECK_TOP + BRG_OD / 2              # roller centre: outer race on the deck
TT_Z0 = ROLL_Z + BRG_OD / 2 + 1.0           # turntable underside (rollers hang from it)
TT_T, TT_R = 6.0, 92.0
TT_Z1 = TT_Z0 + TT_T
ROLL_R = 66.0                               # post face radius; bearing centre ~ r 70
ROLL_ANG = (30, 150, 270)
A = 46.7                                    # tower inner faces at y = +-A
TOW_W = 64.0
ZT = TT_Z1 + 135.0                          # tilt axis height (grip clearance, verified by fit_check.py)
CRADLE_IN, CR_WALL, CR_LEN, CR_WALL_H = 80.0, 5.0, 250.0, 55.0
YI, YO = CRADLE_IN / 2, CRADLE_IN / 2 + CR_WALL
CR_Z0 = ZT + 14.0
CR_Z1 = CR_Z0 + 5.0
TILT_LIMIT = 20.0

# ------------------------------------------------------------------ helpers
def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    b = trimesh.creation.box(hi - lo); b.apply_translation((lo + hi) / 2); return b
def cyl(r, p0, p1, sec=48):
    return trimesh.creation.cylinder(radius=r, segment=[np.asarray(p0, float), np.asarray(p1, float)], sections=sec)
def U(*ms): return trimesh.boolean.union(list(ms), engine="manifold")
def D(a, *bs): return trimesh.boolean.difference([a] + list(bs), engine="manifold")
def W(m, M): m = m.copy(); m.apply_transform(M); return m

# ------------------------------------------------------------------ servo sleeve / plate / floor cap (servo-local)
def sleeve_plate_cuts():
    """Cuts in the plate lying on the servo face (local z 15.9..19.9): horn window + raised-top relief."""
    return [cyl(WIN_R, (SV_AX, 0, SV_FACE - 1), (SV_AX, 0, SV_FACE + PLATE_T + 1)),
            box((-SV_X - C, -9.8, SV_FACE - 0.01), (SV_X + C, 9.8, 17.3))]

def sleeve():
    """Sleeve walls from the plate (local z 15.9) to the open end (local z -15.9), inserts in the open end."""
    outer = box((-SLV_HX, -SLV_HY, -SV_FACE), (SLV_HX, SLV_HY, SV_FACE + 0.01))
    pocket = box((-SV_X - C, -SV_Y - C, -SV_FACE - 1), (SV_X + C, SV_Y + C, SV_FACE + 1))
    ins = [cyl(INSERT_D / 2, (x, y, -SV_FACE - 1), (x, y, -SV_FACE + INSERT_DEPTH), 24) for x, y in INS_XY]
    return D(outer, pocket, *ins)

def floor_cap():
    cap = box((-SLV_HX, -SLV_HY, -SV_FACE - FCAP_T), (SLV_HX, SLV_HY, -SV_FACE))
    relief = box((-SV_X - C, -10.6, -20.4), (SV_X + C, 10.6, -SV_FACE + 1))        # bottom pad + connectors
    cable = box((-8.0, -10.6, -SV_FACE - FCAP_T - 1), (4.0, 10.6, -20.0))           # leads exit through the cap
    holes = []
    for x, y in INS_XY:
        holes.append(cyl(M3_CLEAR / 2, (x, y, -SV_FACE - FCAP_T - 1), (x, y, -SV_FACE + 1), 24))
    return D(cap, relief, cable, *holes)

def hub_and_horn_cuts(top_local):
    """Hub on the horn: local z 18.7 .. top_local. 4x M3x6, 3.0 mm clamp (counterbored), centre-screw recess."""
    hub = cyl(HUB_R, (SV_AX, 0, HORN_FACE), (SV_AX, 0, top_local))
    cuts = [cyl(3.6, (SV_AX, 0, HORN_FACE - 0.5), (SV_AX, 0, HORN_FACE + 2.0))]
    for dx in (-HP, HP):
        for dy in (-HP, HP):
            x, y = SV_AX + dx, dy
            cuts.append(cyl(M3_CLEAR / 2, (x, y, HORN_FACE - 0.5), (x, y, top_local + 1), 24))
            cuts.append(cyl(CB_D / 2, (x, y, HORN_FACE + HORN_CLAMP), (x, y, top_local + 1), 24))
    return hub, cuts

def horn_bolt_cuts(top_local, cb_from_local):
    """4 horn screw holes + counterbores (from cb_from_local up), and the centre-screw recess."""
    cuts = [cyl(3.6, (SV_AX, 0, HORN_FACE - 0.5), (SV_AX, 0, HORN_FACE + 2.0))]
    for dx in (-HP, HP):
        for dy in (-HP, HP):
            x, y = SV_AX + dx, dy
            cuts.append(cyl(M3_CLEAR / 2, (x, y, HORN_FACE - 0.5), (x, y, top_local + 1), 24))
            cuts.append(cyl(CB_D / 2, (x, y, cb_from_local), (x, y, top_local + 1), 24))
    return cuts

# ------------------------------------------------------------------ frames
# pan servo: local z up, output on the world z axis, servo face against the deck underside
PAN_M = Tr((-SV_AX, 0, DECK_Z0 - SV_FACE))
# tilt servo: local z -> world +y (horn points inward), local x -> world +z, local y -> world +x
_R = np.eye(4); _R[:3, 0] = (0, 0, 1); _R[:3, 1] = (1, 0, 0); _R[:3, 2] = (0, 1, 0)
TILT_M = Tr((0, -A - PLATE_T - SV_FACE, ZT)) @ _R @ Tr((-SV_AX, 0, 0))
TILT_HORN_Y = -A - PLATE_T - SV_FACE + HORN_FACE          # -A - 1.3
SPACER_T = (-YO) - TILT_HORN_Y                               # horn face -> ear outer face (3.0)
TILT_EAR_LOCAL = (HORN_FACE + SPACER_T, HORN_FACE + SPACER_T + CR_WALL)
TILT_CLAMP = SPACER_T + 2.5                                  # spacer + 2.5 mm of ear under the head -> M3x8
STUB_BOLTS = [(-13.0, 0.0), (13.0, 0.0), (0.0, -13.0)]   # (x, z) around the tilt axis

# ------------------------------------------------------------------ parts
# servo driver board bay inside the base (static, under the deck, next to the pan servo).
# Sized for boards up to 60 x 40 mm; the board is zip-tied between two ribs.
BOARD_X, BOARD_Y, BOARD_Z = (-30.0, 30.0), (-90.0, -50.0), (26.0, 44.0)

def base():
    deck = box((-100, -100, DECK_Z0), (100, 100, DECK_TOP))
    skirt = D(box((-100, -100, 0), (100, 100, DECK_Z0 + 0.01)), box((-94, -94, -1), (94, 94, DECK_Z0 + 1)))
    feet = [box((sx * 100 - (14 if sx > 0 else 0), sy * 100 - (14 if sy > 0 else 0), 0),
                (sx * 100 + (0 if sx > 0 else 14), sy * 100 + (0 if sy > 0 else 14), DECK_Z0 + 0.01)) for sx in (-1, 1) for sy in (-1, 1)]
    # servo driver board bay: two ribs hanging under the deck, board zip-tied between them
    ribs = [box((x0, BOARD_Y[0] - 2, BOARD_Z[0] - 6), (x0 + 3, BOARD_Y[1] + 2, DECK_Z0 + 0.01))
            for x0 in (BOARD_X[0] - 4, BOARD_X[1] + 1)]
    b = U(deck, skirt, *feet, W(sleeve(), PAN_M), *ribs)
    cuts = [W(c, PAN_M) for c in sleeve_plate_cuts()]
    cuts += [cyl(2.3, (sx * 93, sy * 93, -1), (sx * 93, sy * 93, DECK_TOP + 1)) for sx in (-1, 1) for sy in (-1, 1)]   # M4 desk screws
    cuts += [box((BOARD_X[0], -101, -1), (BOARD_X[1], -93, 36))]                       # 12 V jack + USB + pan lead opening
    cuts += [box((20, -9, DECK_Z0 - 1), (38, 9, DECK_TOP + 1))]                        # cable pass-through from the turntable
    for y in (BOARD_Y[0] + 8, BOARD_Y[1] - 8):                                         # zip-tie holes under the board
        cuts.append(box((BOARD_X[0] - 6, y - 2.5, BOARD_Z[0] - 4.5), (BOARD_X[1] + 6, y + 2.5, BOARD_Z[0] - 1.5)))
    return D(b, *cuts)

def turntable():
    tt = cyl(TT_R, (0, 0, TT_Z0), (0, 0, TT_Z1), 160)
    hub, hcuts = hub_and_horn_cuts(HORN_FACE + (TT_Z1 - PAN_HORN_Z))
    posts = []
    pcuts = []
    for ang in ROLL_ANG:
        Rz = Rot(np.radians(ang), (0, 0, 1))
        p = box((ROLL_R - 12, -8, ROLL_Z - 8), (ROLL_R, 8, TT_Z0 + 0.01)); posts.append(W(p, Rz))
        pcuts.append(W(cyl(INSERT_D / 2, (ROLL_R - INSERT_DEPTH, 0, ROLL_Z), (ROLL_R + 1, 0, ROLL_Z), 24), Rz))
    t = U(tt, W(hub, PAN_M), *posts)
    cuts = [W(c, PAN_M) for c in hcuts] + pcuts
    for sy in (-1, 1):
        for x in (-22, 22):
            for yy in (A + 8, A + 22):
                cuts.append(cyl(INSERT_D / 2, (x, sy * yy, TT_Z1 - INSERT_DEPTH), (x, sy * yy, TT_Z1 + 1), 24))
    # cables (tilt servo, laser, webcam) drop through the +x window into the 25 mm gap, where a slack loop
    # inside r < 50 (clear of the rollers) takes up the pan rotation, then through the deck slot into the base.
    for y in (-66, -58):                                                                 # zip-tie anchor along the cable path
        cuts.append(box((14, y - 2, TT_Z0 - 1), (18, y + 2, TT_Z1 + 1)))
    for x0 in (-52, 30):
        cuts.append(box((x0, -24, TT_Z0 - 1), (x0 + 22, 24, TT_Z1 + 1)))                 # lightening, inside r 60
    return D(t, *cuts)

def roller_axle():
    """flange d17 x 2 (inner race only), stub d14.9 x 5.4, M3 through. Local z = radial outward."""
    return D(U(cyl(8.5, (0, 0, 0), (0, 0, 2.0)), cyl(SHAFT_D / 2, (0, 0, 2.0), (0, 0, 7.4))),
             cyl(M3_CLEAR / 2, (0, 0, -1), (0, 0, 9), 24))

def roller_M(ang):
    return Rot(np.radians(ang), (0, 0, 1)) @ Tr((ROLL_R, 0, ROLL_Z)) @ Rot(np.pi / 2, (0, 1, 0))

def tower_servo():
    """Right tower (y < 0). 4 mm wall = plate over the tilt servo; sleeve integral on the outer face."""
    y0, y1 = -A - PLATE_T, -A
    wall = box((-TOW_W / 2, y0, TT_Z1), (TOW_W / 2, y1, ZT + 22))
    foot = box((-TOW_W / 2, -A - 30, TT_Z1), (TOW_W / 2, y1, TT_Z1 + 6))
    ribs = [box((sx * TOW_W / 2 - (6 if sx > 0 else 0), -A - 30, TT_Z1), (sx * TOW_W / 2 + (0 if sx > 0 else 6), y0 + 0.01, ZT - 50))
            for sx in (-1, 1)]
    t = U(wall, foot, *ribs, W(sleeve(), TILT_M))
    cuts = [W(c, TILT_M) for c in sleeve_plate_cuts()]
    for x in (-22, 22):
        for yy in (A + 8, A + 22):
            cuts.append(cyl(M3_CLEAR / 2, (x, -yy, TT_Z1 - 1), (x, -yy, TT_Z1 + 7), 24))
            cuts.append(cyl(CB_D / 2, (x, -yy, TT_Z1 + 6), (x, -yy, ZT), 24))
    return D(t, *cuts)

def tower_bearing():
    """Left tower (y > 0): 7 mm wall, 6802 seat open to the outside, 1 mm inner lip, foot, ribs."""
    y0, y1 = A, A + 7.0
    wall = box((-TOW_W / 2, y0, TT_Z1), (TOW_W / 2, y1, ZT + 22))
    foot = box((-TOW_W / 2, y0, TT_Z1), (TOW_W / 2, A + 30, TT_Z1 + 6))
    ribs = [box((sx * TOW_W / 2 - (6 if sx > 0 else 0), y1 - 0.01, TT_Z1), (sx * TOW_W / 2 + (0 if sx > 0 else 6), A + 30, ZT - 25))
            for sx in (-1, 1)]
    t = U(wall, foot, *ribs)
    cuts = [cyl(BRG_SEAT / 2, (0, A + 1.0, ZT), (0, A + 8, ZT)), cyl(8.75, (0, A - 1, ZT), (0, A + 1.5, ZT))]
    for x in (-22, 22):
        for yy in (A + 8, A + 22):
            cuts.append(cyl(M3_CLEAR / 2, (x, yy, TT_Z1 - 1), (x, yy, TT_Z1 + 7), 24))
            cuts.append(cyl(CB_D / 2, (x, yy, TT_Z1 + 6), (x, yy, ZT), 24))
    return D(t, *cuts)

def ears_and_cuts():
    """Hinge interface shared by every payload plate: two ears on the tilt axis at y = +-(40..45),
    right ear bolted to the tilt horn through the horn spacer, left ear takes the stub shaft."""
    ears = [U(box((-20, ylo, ZT), (20, yhi, CR_Z0 + 0.01)), cyl(20, (0, ylo, ZT), (0, yhi, ZT)))
            for ylo, yhi in ((YI, YO), (-YO, -YI))]
    cuts = [W(h, TILT_M) for h in horn_bolt_cuts(TILT_EAR_LOCAL[1] + 1, HORN_FACE + TILT_CLAMP)]
    cuts.append(cyl(7.65, (0, YI - 1, ZT), (0, YO + 1, ZT)))
    for x, z in STUB_BOLTS:
        cuts.append(cyl(INSERT_D / 2, (x, YI - 0.5, ZT + z), (x, YO + 0.5, ZT + z), 24))
    return ears, cuts

# laser plate (payload until the Perses arrives)
LP_X = 60.0                         # plate half length
LASER_X0, LASER_X1 = 22.0, 58.0     # laser block along x
LASER_Z = CR_Z1 + 11.0              # bore centre height
CAM_RISER = 20.0                    # webcam pad height, lens clears the laser block

def laser_plate():
    floor = box((-LP_X, -YO, CR_Z0), (LP_X, YO, CR_Z1))
    block = box((LASER_X0, -12, CR_Z1 - 0.01), (LASER_X1, 12, CR_Z1 + 22))
    riser = box((-LP_X, -30, CR_Z1 - 0.01), (-18, 30, CR_Z1 + CAM_RISER))
    ears, cuts = ears_and_cuts()
    p = U(floor, block, riser, *ears)
    cuts += [cyl(10.5 / 2 + 0.15, (LASER_X0 - 1, 0, LASER_Z), (LASER_X1 + 1, 0, LASER_Z)),               # VLM-650-01 bore
             cyl(INSERT_D / 2, (40, 0, CR_Z1 + 22 - INSERT_DEPTH), (40, 0, CR_Z1 + 23), 24),             # set-screw insert
             box((LASER_X0 - 1, -1.5, CR_Z1 - 1), (LASER_X0 + 6, 1.5, LASER_Z)),                          # lead exit slot
             cyl(3.3, (-39, 0, CR_Z0 - 1), (-39, 0, CR_Z1 + CAM_RISER + 1)),                              # 1/4-20 webcam screw
             cyl(6.5, (-39, 0, CR_Z0 - 1), (-39, 0, CR_Z1 + CAM_RISER - 6))]                              # nut / head pocket from below
    for y in (-22, 22):                                                                                   # 20 mm velcro slots for clip webcams
        cuts.append(box((-46, y - 2, CR_Z0 - 1), (-32, y + 2, CR_Z1 + CAM_RISER + 1)))
    cuts.append(box((-12, -30, CR_Z0 - 1), (14, 30, CR_Z1 + 1)))                                          # tool access to the ear screws
    return D(p, *cuts)

def laser_proxy():
    return cyl(10.5 / 2, (LASER_X1 - 19 + 1, 0, LASER_Z), (LASER_X1 + 1, 0, LASER_Z))

def webcam_proxy():
    """Generic USB webcam (C920-sized) sitting on the riser."""
    return box((-52, -47, CR_Z1 + CAM_RISER), (-24, 47, CR_Z1 + CAM_RISER + 30))

def cradle():
    floor = box((-CR_LEN / 2, -YO, CR_Z0), (CR_LEN / 2, YO, CR_Z1))
    walls = [box((-CR_LEN / 2, YI, CR_Z0), (CR_LEN / 2, YO, CR_Z1 + CR_WALL_H)),
             box((-CR_LEN / 2, -YO, CR_Z0), (CR_LEN / 2, -YI, CR_Z1 + CR_WALL_H))]
    ears, cuts = ears_and_cuts()
    c = U(floor, *walls, *ears)
    cuts.append(box((-135, -27, CR_Z0 - 1), (25, 27, CR_Z1 + 1)))                          # grip / battery guard window
    cuts.append(box((37, -YI + 8, CR_Z0 - 1), (CR_LEN / 2 - 12, YI - 8, CR_Z1 + 1)))       # lightening
    for x in (-95, 85):                                                                  # 25 mm strap slots
        cuts.append(box((x - 14, -YO - 1, CR_Z1 + 20), (x + 14, YO + 1, CR_Z1 + 25)))
    c = D(c, *cuts)
    # laser + webcam bracket, front of the right wall
    zt = CR_Z1 + CR_WALL_H
    br = U(box((CR_LEN / 2 - 40, -YO - 30, zt - 6), (CR_LEN / 2, -YO + 0.01, zt)),
           box((CR_LEN / 2 - 30, -YO - 30, zt - 22), (CR_LEN / 2, -YO - 12, zt - 5.99)))
    br = D(br, cyl(10.5 / 2 + 0.15, (CR_LEN / 2 - 31, -YO - 21, zt - 14), (CR_LEN / 2 + 1, -YO - 21, zt - 14)),
           cyl(INSERT_D / 2, (CR_LEN / 2 - 15, -YO - 21, zt - 23), (CR_LEN / 2 - 15, -YO - 21, zt - 17), 24),   # set-screw insert
           cyl(3.3, (CR_LEN / 2 - 20, -YO - 15, zt - 7), (CR_LEN / 2 - 20, -YO - 15, zt + 1)))                   # 1/4-20 webcam screw
    return U(c, br)

def horn_spacer():
    """3 mm disc between the tilt horn and the right ear (world, tilt = 0)."""
    d = cyl(HUB_R, (SV_AX, 0, HORN_FACE), (SV_AX, 0, TILT_EAR_LOCAL[0]))
    return W(D(d, *horn_bolt_cuts(TILT_EAR_LOCAL[0], 99)), TILT_M)

def stub_shaft():
    """Bolt-in tilt shaft: flange d32 x 4 inside the left ear, d14.9 shaft through the ear into the 6802."""
    fl = trimesh.boolean.intersection([cyl(17, (0, YI - 4, ZT), (0, YI, ZT)), box((-20, YI - 5, ZT - 20), (20, YI + 1, CR_Z0 - 2.0))], engine="manifold")
    sh = cyl(SHAFT_D / 2, (0, YI - 0.01, ZT), (0, A + 6.5, ZT))
    cuts = [cyl(INSERT_D / 2, (0, A + 6.5 - INSERT_DEPTH, ZT), (0, A + 7.0, ZT), 24)]
    for x, z in STUB_BOLTS:
        cuts.append(cyl(M3_CLEAR / 2, (x, YI - 5, ZT + z), (x, YI + 0.5, ZT + z), 24))
    return D(U(fl, sh), *cuts)

def shaft_washer():
    return D(cyl(9.0, (0, A + 7.0, ZT), (0, A + 9.0, ZT)), cyl(M3_CLEAR / 2, (0, A + 6, ZT), (0, A + 10, ZT), 24))

def perses_proxy():
    z0 = CR_Z1 + 0.5
    body = box((-P_L / 2 + 40, -P_W / 2, z0), (P_L / 2 - 40, P_W / 2, z0 + P_H))
    hopper = box((-90, -26, z0 + P_H - 1), (210, 26, z0 + P_H + 85))
    barrel = cyl(22, (P_L / 2 - 41, 0, z0 + 70), (P_L / 2 + 20, 0, z0 + 70))
    stock = box((-P_L / 2 - 10, -30, z0 + 15), (-P_L / 2 + 41, 30, z0 + 100))
    grip = W(box((-120, -17, z0 - 115), (-80, 17, z0 + 1)), Rot(np.radians(-12), (0, 1, 0), point=(-100, 0, z0)))
    guard = box((-80, -24, z0 - 60), (10, 24, z0 + 1))
    return U(body, hopper, barrel, stock, grip, guard)

def bearing():
    return D(cyl(BRG_OD / 2, (0, 0, -BRG_W / 2), (0, 0, BRG_W / 2)), cyl(BRG_ID / 2, (0, 0, -3), (0, 0, 3)))

# ------------------------------------------------------------------ build everything (world, pan = tilt = 0)
servo = trimesh.load(SERVO_STL); servo.apply_scale(1000.0)

parts = {
    "base": base(),
    "pan_floor_cap": W(floor_cap(), PAN_M),
    "turntable": turntable(),
    "tower_servo": tower_servo(),
    "tilt_floor_cap": W(floor_cap(), TILT_M),
    "tower_bearing": tower_bearing(),
    "shaft_washer": shaft_washer(),
    "horn_spacer": horn_spacer(),
    "stub_shaft": stub_shaft(),
}
for k, ang in enumerate(ROLL_ANG):
    parts[f"roller_axle_{k+1}"] = W(roller_axle(), roller_M(ang))

# payload on the tilt hinge: "laser" (now) or "perses" (later). TURRET_PAYLOAD=perses to check the cradle.
PAYLOAD = os.environ.get("TURRET_PAYLOAD", "laser")
if PAYLOAD == "perses":
    parts["cradle"] = cradle(); PAYLOAD_HW = {"perses": perses_proxy()}
else:
    parts["laser_plate"] = laser_plate(); PAYLOAD_HW = {"laser": laser_proxy(), "webcam": webcam_proxy()}
    TILT_LIMIT = 40.0

hardware = {"servo_pan": W(servo, PAN_M), "servo_tilt": W(servo, TILT_M), **PAYLOAD_HW,
            "servo_board": box((BOARD_X[0] + 1, BOARD_Y[0] + 1, BOARD_Z[0]), (BOARD_X[1] - 1, BOARD_Y[1] - 1, BOARD_Z[1])),
            "bearing_tilt": W(bearing(), Tr((0, A + 1.0 + BRG_W / 2, ZT)) @ Rot(np.pi / 2, (1, 0, 0)))}
for k, ang in enumerate(ROLL_ANG):
    hardware[f"bearing_roller_{k+1}"] = W(bearing(), roller_M(ang) @ Tr((0, 0, 2.0 + BRG_W / 2 + 0.2)))

# groups for motion
TILTING = {"cradle", "laser_plate", "shaft_washer", "horn_spacer", "stub_shaft"} | set(PAYLOAD_HW)
STATIC = {"base", "pan_floor_cap", "servo_pan", "servo_board"}           # everything else pans

# ------------------------------------------------------------------ screws: (name, part it passes, length, head point, axis, target)
def screws():
    S = []
    # floor caps -> sleeve inserts: cap 8 + 4 into a 5 mm insert = M3x12
    for capname, M in (("pan_floor_cap", PAN_M), ("tilt_floor_cap", TILT_M)):
        for x, y in INS_XY:
            head = (M @ [x, y, -SV_FACE - FCAP_T, 1])[:3]; ax = M[:3, :3] @ [0, 0, 1]
            S.append((f"{capname} -> sleeve insert", 12, head, ax, "insert"))
    # turntable hub -> pan horn, tilt hub -> tilt horn: M3x6, 3.5 clamp
    for label, M, clamp, L in (("turntable hub -> pan horn", PAN_M, HORN_CLAMP, 6), ("cradle ear + spacer -> tilt horn", TILT_M, TILT_CLAMP, 8)):
        for dx in (-HP, HP):
            for dy in (-HP, HP):
                head = (M @ [SV_AX + dx, dy, HORN_FACE + clamp, 1])[:3]; ax = M[:3, :3] @ [0, 0, -1]
                S.append((label, L, head, ax, "horn"))
    # stub shaft flange -> left ear inserts: flange 4 + 4 = M3x8
    for x, z in STUB_BOLTS:
        S.append(("stub flange -> ear insert", 8, np.array([x, YI - 4, ZT + z]), np.array([0, 1.0, 0]), "insert"))
    # tower feet -> turntable inserts: foot 6 + 4 = M3x10
    for sy in (-1, 1):
        for x in (-22, 22):
            for yy in (A + 8, A + 22):
                S.append((f"tower foot -> turntable", 10, np.array([x, sy * yy, TT_Z1 + 6]), np.array([0, 0, -1.0]), "insert"))
    # roller axles -> post inserts: axle 7.4 + 4.6 = M3x12
    for ang in ROLL_ANG:
        M = roller_M(ang)
        S.append(("roller axle -> post insert", 12, (M @ [0, 0, 7.4, 1])[:3], M[:3, :3] @ [0, 0, -1], "insert"))
    if PAYLOAD == "laser":   # laser set screw from the top: M3x6, tip 0.25 mm into the laser body = clamped
        S.append(("laser set screw", 6, np.array([40.0, 0, CR_Z1 + 22]), np.array([0, 0, -1.0]), "insert_setscrew"))
    # shaft washer -> stub insert: 2 + 4 = M3x6
    S.append(("washer -> stub insert", 6, np.array([0, A + 9.0, ZT]), np.array([0, -1.0, 0]), "insert"))
    return S

if __name__ == "__main__":
    if "cradle" not in parts:
        parts_export = dict(parts); parts_export["cradle"] = cradle()
    else:
        parts_export = dict(parts)
    orient = {"base": Rot(np.pi, (1, 0, 0)),                    # deck on the bed, sleeve + skirt up
              "turntable": Rot(np.pi, (1, 0, 0)),               # top on the bed, hub + posts up
              "tower_servo": Rot(-np.pi / 2, (1, 0, 0)),        # inner face on the bed, sleeve up
              "tower_bearing": Rot(np.pi / 2, (1, 0, 0)),       # inner face on the bed
              "cradle": Rot(np.pi, (1, 0, 0)),                   # upside down: wall tops on the bed, ears up
              "laser_plate": Rot(np.pi, (1, 0, 0)),              # upside down: block + riser tops on the bed, ears up
              "shaft_washer": Rot(np.pi / 2, (1, 0, 0)), "horn_spacer": Rot(np.pi / 2, (1, 0, 0)),
              "stub_shaft": Rot(np.pi / 2, (1, 0, 0)),          # flange on the bed, shaft up
              "pan_floor_cap": Rot(np.pi, (1, 0, 0)), "tilt_floor_cap": np.eye(4)}
    for name, m in parts_export.items():
        if name.startswith("roller_axle") and name != "roller_axle_1":
            continue
        mm = m.copy()
        if name.startswith("roller_axle"):
            mm = roller_axle()
        elif name == "tilt_floor_cap":
            mm = floor_cap(); mm.apply_transform(Rot(np.pi, (1, 0, 0)))
        elif name == "pan_floor_cap":
            mm = floor_cap(); mm.apply_transform(Rot(np.pi, (1, 0, 0)))
        else:
            mm.apply_transform(orient.get(name, np.eye(4)))
        mm.apply_translation(-mm.bounds[0])
        fn = {"roller_axle_1": "roller_axle_x3", "pan_floor_cap": "servo_floor_cap_x2"}.get(name, name)
        if name == "tilt_floor_cap":
            continue
        mm.export(os.path.join(OUT, fn + ".stl"))
        print(f"{fn:22s} {np.round(mm.extents,1)}  watertight={mm.is_watertight}  vol={mm.volume/1000:.0f} cm3")
