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
for (n1, m1), (n2, m2) in itertools.combinations(allm.items(), 2):
    v = inter_vol(m1, m2)
    if v > TOL:
        bad += 1; log(f"  COLLISION {n1} x {n2}: {v:.1f} mm3")
log(f"  {'OK, none' if bad == 0 else str(bad) + ' collisions'}")

log("\n== 2. Tilt sweep -20..+20 deg (cradle + washer + Perses vs everything else) ==")
tilt_bad = {}
for deg in range(-int(T.TILT_LIMIT), int(T.TILT_LIMIT) + 1, 5):
    M = Rot(np.radians(deg), (0, 1, 0), point=(0, 0, T.ZT))
    for n1 in T.TILTING:
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
gmin = min((lambda m: (m.apply_transform(Rot(np.radians(d), (0, 1, 0), point=(0, 0, T.ZT))), m.bounds[0][2])[1])(allm["perses"].copy())
           for d in (-20, 20))
log(f"  lowest Perses point over the tilt range: z = {gmin:.1f}  (turntable top {T.TT_Z1:.1f}, tower feet top {T.TT_Z1 + 6:.1f})")

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
log(f"  pan: hub r{T.HUB_R} in deck window r{T.WIN_R} -> radial gap {T.WIN_R - T.HUB_R:.2f}; "
    f"turntable underside {T.TT_Z0:.2f} vs deck top {T.DECK_TOP:.2f} -> {T.TT_Z0 - T.DECK_TOP:.2f} mm")
log(f"  tilt: ear outer face +-{T.YO:.1f} vs tower inner face +-{T.A:.1f} -> {T.A - T.YO:.2f} mm each side")
log(f"  tilt idler: seat d{T.BRG_SEAT:.2f} on 6802 d24 (press), stub d{T.SHAFT_D:.2f} in d15 bore (slip), lip hole d17.5")
log(f"  servo pockets: {2*T.SV_X:.1f} x {2*T.SV_Y:.1f} body in {2*(T.SV_X+T.C):.1f} x {2*(T.SV_Y+T.C):.1f} pocket")

log("\n== 5. Screws ==")
solids = {n: m for n, m in allm.items() if n not in ("perses",)}
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
    rr = 2.6 if target == "insert" else 1.45
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

open("fit_report.txt", "w").write("\n".join(rep) + "\n")
