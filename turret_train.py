"""Train the Tello detector on datasets/tello (made by turret_label.py) and export it for turret_track.py.

    python turret_train.py              # yolo11n, 50 epochs, 640 px, keeps the final epoch
    python turret_train.py --epochs 120 --model yolo11s.pt
    python turret_train.py --review     # no training: run models/tello.pt on the val clip, save runs/review_<name>.jpg
                                        # e.g. --review --weights runs\\detect\\runs\\tello\\weights\\best.pt runs\\detect\\runs\\tello\\weights\\last.pt

Uses the NVIDIA GPU if PyTorch can see one (minutes), else the CPU (an hour or more).
Copies the best weights to models/tello.pt (+ models/tello.onnx).

Settings for this camera: it's mono, so no colour jitter; the lens is fixed on the turret, so no flips up/down,
small rotations only. Brightness jitter stays on (rooms and exposure change).
"""
import argparse, os, shutil, sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join("datasets", "tello", "data.yaml"))
    ap.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt (fastest) or yolo11s.pt (a bit better, ~2x slower)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--review", action="store_true", help="skip training; check models/tello.pt on the val images")
    ap.add_argument("--weights", nargs="+", default=[os.path.join("models", "tello.pt")], help="for --review (one or more)")
    a = ap.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Needs ultralytics:  pip install ultralytics")
    if not os.path.exists(a.data):
        sys.exit(f"No {a.data}. Run turret_label.py on your clips first.")
    gpu = torch.cuda.is_available()
    if a.review:
        for w in a.weights:
            review(YOLO(w), os.path.dirname(a.data), 0 if gpu else "cpu", a.imgsz,
                   "review_" + os.path.splitext(os.path.basename(w))[0])
        return
    print("training on", torch.cuda.get_device_name(0) if gpu else "CPU (slow; an NVIDIA GPU + CUDA PyTorch is ~20x faster)")

    model = YOLO(a.model)
    model.train(data=os.path.abspath(a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch if gpu else 8,
                device=0 if gpu else "cpu", workers=2, project="runs", name="tello", exist_ok=True,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.4, degrees=5.0, flipud=0.0, fliplr=0.5, mosaic=1.0,
                close_mosaic=10, patience=0, plots=True)

    # keep the LAST epoch, not ultralytics' "best": with one held-out clip the val score swings from epoch to epoch and
    # twice picked epoch 1-3, whose scores are too low to use (the last one found 90% at 0.5 vs 13% for "best")
    best = os.path.join(str(model.trainer.save_dir), "weights", "last.pt")   # ultralytics picks the folder (runs/detect/...)
    os.makedirs("models", exist_ok=True)
    shutil.copy(best, os.path.join("models", "tello.pt"))
    m = YOLO(best)
    r = m.val(data=os.path.abspath(a.data), imgsz=a.imgsz, device=0 if gpu else "cpu", split="val", plots=False)
    print(f"\nheld-out clip:  precision {r.box.mp:.2f}  recall {r.box.mr:.2f}  mAP50 {r.box.map50:.2f}  mAP50-95 {r.box.map:.2f}")
    onnx = m.export(format="onnx", imgsz=a.imgsz, dynamic=False, simplify=True)
    shutil.copy(onnx, os.path.join("models", "tello.onnx"))
    print("saved models/tello.pt and models/tello.onnx")
    review(m, os.path.dirname(a.data), 0 if gpu else "cpu", a.imgsz, "review_final")


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def review(model, root, device, imgsz, tag="review"):
    """Per-frame check on the val clip. Scores every detection once, then reports hits / misses / false alarms at several
    confidence cutoffs (an early-epoch model finds the drone but with low scores, so one fixed cutoff can hide it),
    and saves runs/<tag>.jpg with the frames it gets wrong at the best cutoff."""
    import glob
    import cv2
    import numpy as np
    imgs = sorted(glob.glob(os.path.join(root, "images", "val", "*.jpg")))
    rows = []                                            # (path, gt boxes, pred boxes, pred scores)
    for f in imgs:
        im = cv2.imread(f); H, W = im.shape[:2]
        lab = f.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)[:-4] + ".txt"
        gt = []
        for l in (open(lab).read().split("\n") if os.path.exists(lab) else []):
            if l.strip():
                _, cx, cy, w, h = map(float, l.split())
                gt.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        r = model.predict(im, imgsz=imgsz, conf=0.01, device=device, verbose=False)[0]
        rows.append((f, gt, r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()))

    def score(conf):
        hit = miss = fa = 0; bad = []
        for f, gt, pr, pc in rows:
            p = [b for b, c in zip(pr, pc) if c >= conf]; why = []
            for g in gt:
                if max([iou(g, q) for q in p], default=0) >= 0.3: hit += 1
                else: miss += 1; why.append("MISS")
            for q in p:
                if max([iou(g, q) for g in gt], default=0) < 0.3: fa += 1; why.append("FALSE")
            if why: bad.append((f, gt, pr, pc, why))
        return hit, miss, fa, bad

    n = sum(len(g) for _, g, _, _ in rows)
    top = [max([c for b, c in zip(pr, pc) if max([iou(g, b) for g in gt], default=0) >= 0.3], default=0)
           for _, gt, pr, pc in rows if gt]
    print(f"\n{tag}: {len(imgs)} val frames, {n} drones. Score the model gives the real drone: "
          f"median {np.median(top):.2f}, 10% of frames below {np.percentile(top, 10):.2f}")
    best = None
    for conf in (0.05, 0.1, 0.2, 0.35, 0.5):
        hit, miss, fa, bad = score(conf)
        f1 = 2 * hit / max(2 * hit + miss + fa, 1)
        print(f"  cutoff {conf:.2f}:  found {hit}/{n} ({100 * hit / max(n, 1):.0f}%)  missed {miss}  false alarms {fa}  F1 {f1:.2f}")
        if best is None or f1 > best[0]: best = (f1, conf, bad)
    f1, conf, bad = best
    print(f"  best cutoff {conf:.2f}")
    tiles = []
    for f, gt, pr, pc, why in bad[:60]:
        v = cv2.imread(f)
        for g in gt: cv2.rectangle(v, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), 3)
        for q, c in zip(pr, pc):
            if c < conf: continue
            cv2.rectangle(v, (int(q[0]), int(q[1])), (int(q[2]), int(q[3])), (0, 0, 255), 3)
            cv2.putText(v, f"{c:.2f}", (int(q[0]), int(q[1]) - 6), 0, 1.0, (0, 0, 255), 2)
        cv2.putText(v, os.path.basename(f)[5:-4] + "  " + " ".join(sorted(set(why))), (10, 40), 0, 1.1, (0, 0, 255), 3)
        tiles.append(cv2.resize(v, (427, 240)))
    if tiles:
        while len(tiles) % 5: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(os.path.join("runs", tag + ".jpg"), np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]))
        print(f"  frames it gets wrong at {conf:.2f}: runs/{tag}.jpg  (green = your label, red = model + score)")

if __name__ == "__main__":
    main()
