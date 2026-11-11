"""Label drone clips with the trained detector; you only look at the frames it isn't sure about.

    python turret_autolabel.py recordings\\clip_*.mp4          new clips -> datasets\\tello
    python turret_autolabel.py --fix                           re-check the held-out (val) labels in datasets\\tello
    python turret_autolabel.py --fix --split all               ... or every label
                                                               (the old tracker-made labels drifted in places)

For each frame (every 3rd of a clip) the detector looks for the drone:
  score >= 0.6 and only one strong box   -> labelled automatically
  anything else                          -> shown to you
With --fix, a frame is shown only when the model and the existing label disagree.

Keys on a shown frame (red = model + score, green = existing label):
  1 / Enter   the red box is right (the higher score, marked 1)   3   the other red box (marked 3)
  2           keep the green (existing) label
  N           no drone in this frame        D   draw the box yourself (drag, Enter)
  S           skip (don't use this frame)   B   back one frame      Q   stop (everything so far is saved)
"""
import argparse, glob, os, sys
import cv2
import numpy as np

HI = 0.6


def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    w, h = min(ax1, bx1) - max(a[0], b[0]), min(ay1, by1) - max(a[1], b[1])
    i = max(0, w) * max(0, h)
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def read_label(path, W, H):
    if not os.path.exists(path): return None
    for l in open(path).read().split("\n"):
        if l.strip():
            _, cx, cy, w, h = map(float, l.split())
            return ((cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H)
    return ()                                            # empty file = no drone


def write_label(path, box, W, H):
    with open(path, "w") as f:
        if box:
            x, y, w, h = box
            f.write(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--fix", action="store_true", help="re-check datasets\\tello's existing labels instead")
    ap.add_argument("--split", default="val", choices=["val", "train", "all"],
                    help="with --fix: which labels to re-check (default val, the held-out clip the score comes from)")
    ap.add_argument("--weights", default=os.path.join("models", "tello.pt"))
    ap.add_argument("--out", default=os.path.join("datasets", "tello"))
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--val-every", type=int, default=5, help="every Nth new clip goes to val as a whole")
    a = ap.parse_args()
    import torch
    from ultralytics import YOLO
    model = YOLO(a.weights); dev = 0 if torch.cuda.is_available() else "cpu"

    def detect(img):
        r = model.predict(img, imgsz=640, conf=0.05, device=dev, verbose=False, max_det=5)[0]
        return sorted([((float(x0), float(y0), float(x1 - x0), float(y1 - y0)), float(c))
                       for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())],
                      key=lambda d: -d[1])

    # ---- frames, one at a time: (image, path the image lives at / goes to, label path)
    def frames():
        if a.fix:
            for img_p in sorted(glob.glob(os.path.join(a.out, "images", "*" if a.split == "all" else a.split, "*.jpg"))):
                yield cv2.imread(img_p), img_p, img_p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)[:-4] + ".txt"
            return
        clips = [f for pat in a.clips for f in glob.glob(pat)]
        if not clips: sys.exit("No clips given (or use --fix).")
        for sub in ("images/train", "images/val", "labels/train", "labels/val"):
            os.makedirs(os.path.join(a.out, sub), exist_ok=True)
        for ci, clip in enumerate(clips):
            split = "val" if a.val_every and ci % a.val_every == a.val_every - 1 else "train"
            name = os.path.splitext(os.path.basename(clip))[0]
            cap = cv2.VideoCapture(clip); i = 0
            print(f"{clip} -> {split}")
            while True:
                ok, fr = cap.read()
                if not ok: break
                if i % a.every == 0:
                    stem = f"{name}_{i:06d}"
                    yield fr, os.path.join(a.out, "images", split, stem + ".jpg"), os.path.join(a.out, "labels", split, stem + ".txt")
                i += 1
            cap.release()

    # ---- pass 1: label what's clear, queue the rest (queued clip frames are saved now, labelled in pass 2)
    queue, ok_n, n = [], 0, 0
    for img, img_p, lab_p in frames():
        n += 1
        H, W = img.shape[:2]
        dets = detect(img)
        strong = [d for d in dets if d[1] >= HI]
        clear = len(strong) == 1 and not [d for d in dets[1:] if d[1] >= 0.35]
        if a.fix:
            old = read_label(lab_p, W, H)
            agree = (old == () and not [d for d in dets if d[1] >= 0.35]) or (old and clear and iou(old, strong[0][0]) >= 0.5)
            if agree: ok_n += 1; continue
        else:
            old = None
            cv2.imwrite(img_p, img)
            if clear: write_label(lab_p, strong[0][0], W, H); ok_n += 1; continue
        queue.append((img_p, lab_p, dets, old))
        if n % 200 == 0: print(f"  {n} frames checked")
    print(f"{ok_n} of {n} frames OK by themselves, {len(queue)} for you to look at")

    # ---- pass 2: you decide the unclear ones
    win = "autolabel"
    if queue: cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    k_i, done = 0, 0
    while 0 <= k_i < len(queue):
        img_p, lab_p, dets, old = queue[k_i]
        if not os.path.exists(img_p) and os.path.exists(img_p + ".skipped"):   # came back (B) to a skipped frame
            os.replace(img_p + ".skipped", img_p)
            if os.path.exists(lab_p + ".skipped"): os.replace(lab_p + ".skipped", lab_p)
        img = cv2.imread(img_p)
        H, W = img.shape[:2]; v = img.copy()
        for b, c in dets:
            x, y, w, h = map(int, b); cv2.rectangle(v, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(v, f"{c:.2f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(dets) > 1:                                # number the top two so 1 / 3 are clear
            for j, (b, c) in enumerate(dets[:2]):
                cv2.putText(v, "1" if j == 0 else "3", (int(b[0] + b[2]) + 4, int(b[1] + b[3])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if old:
            x, y, w, h = map(int, old); cv2.rectangle(v, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(v, f"{k_i + 1}/{len(queue)}  1/Enter red  3 2nd red  2 green  N none  D draw  S skip  B back  Q quit",
                    (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(win, v)
        k = cv2.waitKey(0) & 0xFF
        box = None
        if k in (ord("1"), 13) and dets: box = dets[0][0]
        elif k == ord("3") and len(dets) > 1: box = dets[1][0]
        elif k == ord("2") and old is not None: box = old
        elif k == ord("n"): box = ()
        elif k == ord("d"):
            r = cv2.selectROI(win, img, showCrosshair=True)
            if r[2] > 3 and r[3] > 3: box = tuple(float(x) for x in r)
        elif k == ord("b"): k_i = max(0, k_i - 1); continue
        elif k == ord("q"): break
        elif k == ord("s"):                              # not used for training: set the frame aside
            for p in (img_p, lab_p):
                if os.path.exists(p): os.replace(p, p + ".skipped")
            k_i += 1; continue
        if box is None: continue                         # key that doesn't apply here: ask again
        write_label(lab_p, box, W, H); done += 1; k_i += 1
    if not a.fix:                                        # clip frames you never got to: keep them out of training
        for img_p, lab_p, _, _ in queue[k_i:]:
            if not os.path.exists(lab_p) and os.path.exists(img_p): os.replace(img_p, img_p + ".skipped")
    if queue: cv2.destroyAllWindows()
    print(f"{done} frames labelled by you. Retrain:  turret_train.bat")


if __name__ == "__main__":
    main()
