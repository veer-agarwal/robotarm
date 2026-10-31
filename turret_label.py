"""Turn a recorded clip into YOLO training data: box the drone once, the tracker follows it,
and every few frames an image + label is saved.

    python turret_label.py recordings\\clip_20261031_141500.mp4
    python turret_label.py recordings\\*.mp4 --every 3

Keys while it runs:
  SPACE  pause and redraw the box (do this whenever the green box drifts off the drone)
  N      the drone is not in view: skip frames until you redraw (SPACE)
  Q      stop this clip (what was saved is kept)

Output: datasets\\tello\\images\\{train,val}\\*.jpg + labels\\{train,val}\\*.txt (class 0 = drone) and
datasets\\tello\\data.yaml, ready for:  yolo train model=yolo11n.pt data=datasets/tello/data.yaml epochs=80
"""
import argparse, glob, os, sys
import cv2
from turret_track import make_tracker


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", default=os.path.join("datasets", "tello"))
    ap.add_argument("--every", type=int, default=3, help="save every Nth frame")
    a = ap.parse_args()
    clips = [f for pat in a.clips for f in glob.glob(pat)]
    if not clips: sys.exit("No clips found.")
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    saved = 0
    win = "label"; cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for clip in clips:
        cap = cv2.VideoCapture(clip); name = os.path.splitext(os.path.basename(clip))[0]
        tracker, i, paused_absent = None, 0, False
        print(f"{clip}: draw a box around the drone, Enter. (No drone in this frame? press C in the box window to cancel, then N)")
        while True:
            ok, frame = cap.read()
            if not ok: break
            H, W = frame.shape[:2]; view = frame.copy()
            if tracker is None and not paused_absent:
                r = cv2.selectROI(win, frame, showCrosshair=True)
                if r[2] > 3 and r[3] > 3:
                    tracker, _ = make_tracker(); tracker.init(frame, r)
                else:
                    paused_absent = True
            box = None
            if tracker is not None:
                found, b = tracker.update(frame)
                if found:
                    box = [int(v) for v in b]
                    cv2.rectangle(view, box[:2], (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                else:
                    cv2.putText(view, "lost - SPACE to redraw", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if i % a.every == 0 and (box is not None or paused_absent):
                split = "val" if (saved % 10 == 9) else "train"
                stem = f"{name}_{i:06d}"
                cv2.imwrite(os.path.join(a.out, "images", split, stem + ".jpg"), frame)
                with open(os.path.join(a.out, "labels", split, stem + ".txt"), "w") as f:
                    if box is not None:          # empty label file = a "no drone here" example, also useful
                        x, y, w, h = box
                        f.write(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
                saved += 1
            cv2.putText(view, f"{name}  frame {i}  saved {saved}" + ("  [no drone]" if paused_absent else ""),
                        (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            if k == ord(" "): tracker = None; paused_absent = False
            if k == ord("n"): tracker = None; paused_absent = True
            i += 1
        cap.release()
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n")
    cv2.destroyAllWindows()
    print(f"saved {saved} labeled frames to {a.out}")


if __name__ == "__main__":
    main()
