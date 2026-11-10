"""Train the Tello detector on datasets/tello (made by turret_label.py) and export it for turret_track.py.

    python turret_train.py              # yolo11n, 80 epochs, 640 px
    python turret_train.py --epochs 120 --model yolo11s.pt
    python turret_train.py --review     # no training: run models/tello.pt on the val clip, save runs/review.jpg
                                        # (green = your label, red = what the model found, only frames it got wrong)

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
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--review", action="store_true", help="skip training; check models/tello.pt on the val images")
    ap.add_argument("--weights", default=os.path.join("models", "tello.pt"), help="for --review")
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
        return review(YOLO(a.weights), os.path.dirname(a.data), 0 if gpu else "cpu", a.imgsz)
    print("training on", torch.cuda.get_device_name(0) if gpu else "CPU (slow; an NVIDIA GPU + CUDA PyTorch is ~20x faster)")

    model = YOLO(a.model)
    model.train(data=os.path.abspath(a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch if gpu else 8,
                device=0 if gpu else "cpu", workers=2, project="runs", name="tello", exist_ok=True,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.4, degrees=5.0, flipud=0.0, fliplr=0.5, mosaic=1.0,
                close_mosaic=10, patience=25, plots=True)

    best = os.path.join(str(model.trainer.save_dir), "weights", "best.pt")   # ultralytics picks the folder (runs/detect/...)
    os.makedirs("models", exist_ok=True)
    shutil.copy(best, os.path.join("models", "tello.pt"))
    m = YOLO(best)
    r = m.val(data=os.path.abspath(a.data), imgsz=a.imgsz, device=0 if gpu else "cpu", split="val", plots=False)
    print(f"\nheld-out clip:  precision {r.box.mp:.2f}  recall {r.box.mr:.2f}  mAP50 {r.box.map50:.2f}  mAP50-95 {r.box.map:.2f}")
    onnx = m.export(format="onnx", imgsz=a.imgsz, dynamic=False, simplify=True)
    shutil.copy(onnx, os.path.join("models", "tello.onnx"))
    print("saved models/tello.pt and models/tello.onnx")
    review(m, os.path.dirname(a.data), 0 if gpu else "cpu", a.imgsz)


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def review(model, root, device, imgsz, conf=0.35):
    """Per-frame check on the val clip: hit / miss / false alarm, misses by drone size, and a sheet of the bad frames."""
    import glob
    import cv2
    import numpy as np
    imgs = sorted(glob.glob(os.path.join(root, "images", "val", "*.jpg")))
    hit = miss = fa = loose = 0; bad = []; sizes = {"hit": [], "miss": []}
    for f in imgs:
        im = cv2.imread(f); H, W = im.shape[:2]
        lab = f.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)[:-4] + ".txt"
        gt = []
        for l in open(lab).read().split("\n") if os.path.exists(lab) else []:
            if l.strip():
                _, cx, cy, w, h = map(float, l.split())
                gt.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        r = model.predict(im, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        pr = r.boxes.xyxy.cpu().numpy().tolist(); pc = r.boxes.conf.cpu().numpy().tolist()
        why = []
        for g in gt:
            best = max([iou(g, p) for p in pr], default=0)
            if best >= 0.3:
                hit += 1; sizes["hit"].append(g[2] - g[0])
                if best < 0.5: loose += 1
            else:
                miss += 1; sizes["miss"].append(g[2] - g[0]); why.append("MISS")
        for p in pr:
            if max([iou(g, p) for g in gt], default=0) < 0.3: fa += 1; why.append("FALSE")
        if why:
            v = im.copy()
            for g in gt: cv2.rectangle(v, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), 3)
            for p, c in zip(pr, pc):
                cv2.rectangle(v, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 0, 255), 3)
                cv2.putText(v, f"{c:.2f}", (int(p[0]), int(p[1]) - 6), 0, 1.0, (0, 0, 255), 2)
            cv2.putText(v, os.path.basename(f)[5:-4] + "  " + " ".join(sorted(set(why))), (10, 40), 0, 1.1, (0, 0, 255), 3)
            bad.append(cv2.resize(v, (427, 240)))
    n = hit + miss
    print(f"\nval clip, conf {conf}:  found {hit}/{n} drones ({100 * hit / max(n, 1):.0f}%), missed {miss}, "
          f"false alarms {fa} in {len(imgs)} frames, loose boxes {loose}")
    for k in ("hit", "miss"):
        if sizes[k]: print(f"  {k:4s} drone width px: median {np.median(sizes[k]):.0f}  min {min(sizes[k]):.0f}  max {max(sizes[k]):.0f}")
    if bad:
        bad = bad[:60]
        while len(bad) % 5: bad.append(np.zeros_like(bad[0]))
        sheet = np.vstack([np.hstack(bad[i:i + 5]) for i in range(0, len(bad), 5)])
        cv2.imwrite(os.path.join("runs", "review.jpg"), sheet)
        print("frames it got wrong: runs/review.jpg  (green = your label, red = model)")


if __name__ == "__main__":
    main()
