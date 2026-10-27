"""One-file assembly with every part as its own named body: turret_assembly.3mf + .glb (pan 0, tilt 0).
Printed parts keep their names, bought hardware is prefixed HW_. TURRET_PAYLOAD=perses for the cradle version."""
import sys
sys.argv.append("--no-export")
import numpy as np, trimesh, matplotlib.colors as mc
import perses_turret as T
from render import COL

sc = trimesh.Scene()
for name, m in {**T.parts, **T.hardware}.items():
    mm = m.copy()
    hw = name in T.hardware
    c = np.array(mc.to_rgba(COL.get(name, "#d4af37" if "bearing" in name else "#5b6470"))) * 255
    if name == "future_pi5": c[3] = 110
    mm.visual = trimesh.visual.ColorVisuals(mm, face_colors=np.tile(c.astype(np.uint8), (len(mm.faces), 1)))
    sc.add_geometry(mm, node_name=("HW_" if hw else "") + name, geom_name=("HW_" if hw else "") + name)
suffix = "" if T.PAYLOAD == "laser" else "_perses"
sc.export(f"turret_assembly{suffix}.3mf"); sc.export(f"turret_assembly{suffix}.glb")
print(len(sc.geometry), "bodies ->", f"turret_assembly{suffix}.3mf/.glb")
