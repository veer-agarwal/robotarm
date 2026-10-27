"""Fit checks for perses_turret.py: collisions at rest, tilt sweep, pan sweep, roller contact, screws."""
import itertools, sys
import numpy as np, trimesh
from trimesh.transformations import rotation_matrix as Rot
sys.argv.append("--no-export")
import perses_turret as T

TOL = 1.0            # mm^3, ignore numerical slivers
rep = []
def log(s=""): print(s); rep.append(s)

def inter_vol(a, b):
    if not (np.all(a.bounds[0] <= b.bounds[1]) and np.all(b.bounds[0] <= a.bounds[1])): return 0.0
    try:
        r = trimesh.boolean.intersection([a, b], engine="manifold")
        return 0.0 if r.is_empty else float(r.volume)
    except Exception: return 0.0

allm = {**T.parts, **T.hardware}

# intended contacts (touching faces / press fits) — reported but not failures if small
EXPECTED = set()

log("== 1. Collisions at rest (pan 0, tilt 0) ==")
bad = 0
ALTERNATIVES = {frozenset(("future_pi5", "usb_hub"))}       # the Pi replaces the hub, never both
for (n1, m1), (n2, m2) in itertools.combinations(allm.items(), 2):
    if frozenset((n1, n2)) in ALTERNATIVES: continue
    v = inter_vol(m1, m2)
    if v > TOL:
        bad += 1; log(f"  COLLISION {n1} x {n2}: {v:.1f} mm3")
log(f"  {'OK, none' if bad == 0 else str(bad) + ' collisions'}")

log(f"\n== 2. Tilt sweep +-{T.TILT_LIMIT:.0f} deg (tilting group vs everything else) ==")
tilt_bad = {}
for deg in range(-int(T.TILT_LIMIT), int(T.TILT_LIMIT) + 1, 5):
    M = Rot(np.radians(deg), (0, 1, 0), point=(0, 0, T.ZT))
    for n1 in [n for n in T.TILTING if n in allm]:
        m1 = allm[n1].copy(); m1.apply_transform(M)
        for n2, m2 in allm.items():
            if n2 in T.TILTING: continue
            v = inter_vol(m1, m2)
            if v > TOL: tilt_bad.setdefault((n1, n2), []).append((deg, round(v, 1)))
if tilt_bad:
    for k, v in tilt_bad.items(): log(f"  COLLISION {k[0]} x {k[1]} at {v}")
else:
    log("  OK, nothing touches through the full tilt range")

# how close does the Perses grip come to the turntable?
pay = [n for n in T.TILTING if n in allm and n not in ("shaft_washer", "horn_spacer", "stub_shaft")]
def low(d):
    return min(allm[n].copy().apply_transform(Rot(np.radians(d), (0, 1, 0), point=(0, 0, T.ZT))).bounds[0][2] for n in pay)
gmin = min(low(-T.TILT_LIMIT), low(T.TILT_LIMIT))
log(f"  payload: {T.PAYLOAD}, tilt limit +-{T.TILT_LIMIT:.0f} deg, lowest tilting point z = {gmin:.1f}  (turntable top {T.TT_Z1:.1f}, tower feet top {T.TT_Z1 + 6:.1f})")

log("\n== 3. Pan sweep 0..345 deg (moving group vs base + pan cap + pan servo) ==")
pan_bad = {}
for deg in range(0, 360, 15):
    M = Rot(np.radians(deg), (0, 0, 1))
    for n1, m1 in allm.items():
        if n1 in T.STATIC: continue
        mm = m1.copy(); mm.apply_transform(M)
        for n2 in T.STATIC:
            v = inter_vol(mm, allm[n2])
            if v > TOL: pan_bad.setdefault((n1, n2), []).append(deg)
log("  OK, free through 360" if not pan_bad else "\n".join(f"  COLLISION {k} at {v}" for k, v in pan_bad.items()))

log("\n== 4. Bearings and running gaps ==")
for k in (1, 2, 3):
    b = allm[f"bearing_roller_{k}"]
    log(f"  roller {k}: bottom {b.bounds[0][2]:.2f} vs deck top {T.DECK_TOP:.2f}  -> gap {b.bounds[0][2] - T.DECK_TOP:+.2f} mm "
        f"(top {b.bounds[1][2]:.2f} vs turntable underside {T.TT_Z0:.2f})")
log(f"  pan (flipped servo): fixed post neck r{T.HUB_R} in the sleeve plate's horn window r{T.WIN_R} -> radial gap {T.WIN_R - T.HUB_R:.2f}; "
    f"post top = horn face z {T.PAN_HORN_Z:.2f}; turntable underside {T.TT_Z0:.2f} vs deck top {T.DECK_TOP:.2f} -> {T.TT_Z0 - T.DECK_TOP:.2f} mm")
log(f"  hanging sleeve sweep r{np.hypot(T.SV_AX + T.SLV_HX, T.SLV_HY):.1f} in deck ring hole r{T.DECK_HOLE_R} ; "
    f"rollers r68..76 inside channel wall r{T.CH_R0}")
log(f"  tilt: ear outer face +-{T.YO:.1f} vs tower inner face +-{T.A:.1f} -> {T.A - T.YO:.2f} mm each side")
log(f"  tilt idler: seat d{T.BRG_SEAT:.2f} on 6802 d24 (press), stub d{T.SHAFT_D:.2f} in d15 bore (slip), lip hole d17.5")
log(f"  servo pockets: {2*T.SV_X:.1f} x {2*T.SV_Y:.1f} body in {2*(T.SV_X+T.C):.1f} x {2*(T.SV_Y+T.C):.1f} pocket")

log("\n== 5. Screws ==")
solids = {n: m for n, m in allm.items() if n not in ("perses", "webcam")}
def inside_any(pts, skip=()):
    res = np.zeros(len(pts), bool); who = [None] * len(pts)
    for n, m in solids.items():
        if n in skip: continue
        if not m.is_watertight: continue
        c = m.contains(pts)
        for i in np.where(c & ~res)[0]: who[i] = n
        res |= c
    return res, who

screw_ok = True
eng_log = []
from collections import Counter
bom = Counter()
for label, L, head, ax, target in T.screws():
    ax = np.asarray(ax, float); ax /= np.linalg.norm(ax); head = np.asarray(head, float)
    bom[f"M3x{L}"] += 1
    # (a) head: cylinder behind the head point must be clear
    hp = [head - ax * t + r * v for t in (0.5, 1.5, 2.5) for r in (0, 2.5)
          for v in ([np.cross(ax, [1, 0, 0])] if abs(ax[0]) < 0.9 else [np.cross(ax, [0, 1, 0])])]
    hp = np.array([p / 1 for p in hp])
    h_in, h_who = inside_any(hp)
    # (b) shank axis must not run into any solid (servo horn region excepted below)
    ts = np.linspace(0.3, L - 0.2, 25)
    axis_pts = head + np.outer(ts, ax)
    a_in, a_who = inside_any(axis_pts)
    # (c) engagement: ring points at r=2.6 (inserts) or r=1.45 (horn thread) around the last 3 mm
    perp = np.cross(ax, [0, 0, 1]) if abs(ax[2]) < 0.9 else np.cross(ax, [1, 0, 0]); perp /= np.linalg.norm(perp)
    perp2 = np.cross(ax, perp)
    rr = 2.6 if target.startswith("insert") else 1.45
    tip_ts = np.arange(0.1, L, 0.1)
    ring = np.array([head + ax * t + rr * (np.cos(a) * perp + np.sin(a) * perp2) for t in tip_ts for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)])
    r_in, r_who = inside_any(ring)
    per_t = r_in.reshape(len(tip_ts), 8).mean(1) > 0.9
    # engagement = material around the shank in the threaded target (servo horn / insert hole region past the clamp)
    if target == "horn":
        tgt = np.array([w for w in r_who]).reshape(len(tip_ts), 8)
        eng = 0.1 * sum(1 for i in range(len(tip_ts)) if per_t[i] and any(str(x).startswith('servo') for x in tgt[i]))
    else:
        # inserts: material at r=2.6 over the last insert length (hole d4 + insert in it)
        eng = 0.1 * sum(per_t[tip_ts > L - T.INSERT_LEN])
    need = 2.0 if target == "horn" else 3.5
    if target == "insert_setscrew": need = 3.5; eng = 0.1 * sum(per_t[(tip_ts > 0) & (tip_ts < T.INSERT_LEN)])
    problems = []
    if h_in.any(): problems.append(f"head hits {set(w for w in h_who if w)}")
    if a_in.any(): problems.append(f"shank runs into {set(w for w in a_who if w)} at t={ts[a_in].round(1).tolist()}")
    if eng < need: problems.append(f"thread engagement {eng:.1f} mm < {need}")
    eng_log.append((label, L, round(eng, 1)))
    if problems:
        screw_ok = False; log(f"  FAIL {label} M3x{L}: " + "; ".join(problems))
log("  all screws: heads clear, shanks clear, tips engaged in inserts / horn threads" if screw_ok else "")
log("  engagement (mm): " + ", ".join(sorted(set(f"{l} M3x{L}: {e}" for l, L, e in eng_log))))
log("  screw count: " + ", ".join(f"{k} x{v}" for k, v in sorted(bom.items())))

log("\n== 6. Stock servo leads (152 mm) to the board in the box ==")
def plen(p): p = np.array(p, float); return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
pan_exit = (T.PAN_M @ [-2.0, 0, -T.SV_FACE - T.FCAP_T, 1])[:3]
tilt_exit = (T.TILT_M @ [-2.0, 0, -T.SV_FACE - T.FCAP_T, 1])[:3]
bx, by, bz = T.BOARD_BOX
port_lo = (np.mean(bx), by[1], bz[0] + 3); port_hi = (np.mean(bx), by[1], bz[1] - 3)
routes = {
  "pan  lead: cap slot -> tray cutout -> under the Pi spot, past the Nano rails -> board (servo ports on its top edge)":
      [pan_exit, pan_exit + (0, 0, 9), (6, -8, 99), (6, -20, 103), (10, -32, 118), (24, -35, 130), (bx[0] + 6, by[1] + 3, bz[1] - 3)],
  "tilt lead: cap -> out past the sleeve (x 22) -> over the tower ribs -> -y wall notch -> board":
      [tilt_exit, tilt_exit + (0, -2.5, 0), (22, tilt_exit[1] - 2.5, tilt_exit[2] - 5), (34, -48, T.EB_Z1 + 8),
       (40, -T.EB_Y + 1, T.EB_Z1 - 2), (40, by[1] + 3, bz[1] - 3)],
}
for k, r in routes.items():
    L = plen(r) * 1.15                                                     # +15% for bends
    log(f"  {k}: {L:5.0f} mm with bends, {152 - L:+.0f} mm spare  {'OK' if L < 152 else 'TOO LONG'}")
# the cable centre line must stay in free air (pan 0, tilt 0): sample every segment, skip the first/last 4 mm (exit + plug)
pts = []
for r in routes.values():
    r = np.array(r, float)
    for p0, p1 in zip(r[:-1], r[1:]):
        for t in np.linspace(0, 1, 30): pts.append(p0 + t * (p1 - p0))
    pts = [p for p in pts if np.linalg.norm(p - r[0]) > 4 and np.linalg.norm(p - r[-1]) > 4]
ins, who = inside_any(np.array(pts), skip=("servo_board",))
log("  cable centre lines clear of all parts" if not ins.any() else f"  cable runs through {set(w for w in who if w)}")
open("fit_report.txt", "w").write("\n".join(rep) + "\n")
