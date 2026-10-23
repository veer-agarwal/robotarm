"""
Live camera viewer. Shows each camera index in its own window so you can focus the
lens and check which index is which. Press Q in any window to quit.

Usage:
    python cams.py            # tries indices 0-3 and shows whatever opens
    python cams.py 0 1        # show specific indices
"""

import sys

import cv2

W, H, FPS = 640, 480, 30


def main() -> None:
    wanted = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3]
    caps = {}
    for i in wanted:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DirectShow is the reliable backend on Windows
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        ok, _ = cap.read()
        if ok:
            caps[i] = cap
            print(f"camera {i}: open, {int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5):.0f} fps")
        else:
            cap.release()
    if not caps:
        raise SystemExit("No cameras opened. Try different indices, e.g. python cams.py 0 1 2 3 4 5")

    print("Twist the wrist camera lens until the image is sharp. Press Q to quit.")
    while True:
        for i, cap in caps.items():
            ok, frame = cap.read()
            if ok:
                cv2.putText(frame, f"index {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f"camera {i}", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
