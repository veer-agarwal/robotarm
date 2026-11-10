"""Train the Tello detector on datasets/tello (made by turret_label.py) and export it for turret_track.py.

    python turret_train.py              # yolo11n, 80 epochs, 640 px
    python turret_train.py --epochs 120 --model yolo11s.pt

Uses the NVIDIA GPU if PyTorch can see one (minutes), else the CPU (an hour or more).
Writes runs/tello/weights/best.pt and copies it to models/tello.pt (+ models/tello.onnx).

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
    a = ap.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Needs ultralytics:  pip install ultralytics")
    if not os.path.exists(a.data):
        sys.exit(f"No {a.data}. Run turret_label.py on your clips first.")
    gpu = torch.cuda.is_available()
    print("training on", torch.cuda.get_device_name(0) if gpu else "CPU (slow; an NVIDIA GPU + CUDA PyTorch is ~20x faster)")

    model = YOLO(a.model)
    model.train(data=os.path.abspath(a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch if gpu else 8,
                device=0 if gpu else "cpu", workers=2, project="runs", name="tello", exist_ok=True,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.4, degrees=5.0, flipud=0.0, fliplr=0.5, mosaic=1.0,
                close_mosaic=10, patience=25, plots=True)

    best = os.path.join("runs", "tello", "weights", "best.pt")
    os.makedirs("models", exist_ok=True)
    shutil.copy(best, os.path.join("models", "tello.pt"))
    m = YOLO(best)
    r = m.val(data=os.path.abspath(a.data), imgsz=a.imgsz, device=0 if gpu else "cpu", split="val", plots=False)
    print(f"\nheld-out clip:  precision {r.box.mp:.2f}  recall {r.box.mr:.2f}  mAP50 {r.box.map50:.2f}  mAP50-95 {r.box.map:.2f}")
    onnx = m.export(format="onnx", imgsz=a.imgsz, dynamic=False, simplify=True)
    shutil.copy(onnx, os.path.join("models", "tello.onnx"))
    print("saved models/tello.pt and models/tello.onnx")


if __name__ == "__main__":
    main()
