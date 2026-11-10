"""Open a camera on Windows without guessing the backend.

DirectShow is usually the best choice, but on some PCs it can't open anything ("can't be used to capture by index")
while Media Foundation (MSMF) works. open_capture tries them in order and keeps the first that delivers a frame.

MSMF quirks handled here: it opens slowly unless hardware transforms are off (set below), and asking it for MJPG or
changing the size after opening can break it ("Failed to select stream", then a crash on read), so on MSMF the size
and frame rate are passed when the camera is opened and the format is left to the driver.
"""
import os
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")   # MSMF opens in ~1 s instead of ~8 s
import cv2

ORDER = ["DSHOW", "MSMF", "ANY"] if os.name == "nt" else ["ANY"]
_last_good = None


def _works(cap):
    try:
        ok, f = cap.read()
        return bool(ok) and f is not None
    except cv2.error:
        return False


def open_capture(idx, w=None, h=None, fps=None, mjpg=True, prefer=None, quiet=False):
    """Return (cap, backend_name), or (None, None). w/h/fps are requests; check cap.get() for what you got."""
    global _last_good
    order = list(ORDER)
    for first in (prefer, _last_good):
        if first and first in order:
            order.remove(first); order.insert(0, first)
    for name in order:
        api = getattr(cv2, f"CAP_{name}")
        tries = []
        if name == "MSMF" and w and h:
            params = [cv2.CAP_PROP_FRAME_WIDTH, w, cv2.CAP_PROP_FRAME_HEIGHT, h]
            if fps: params += [cv2.CAP_PROP_FPS, fps]
            tries.append(params)
        tries.append(None)
        for params in tries:
            try:
                cap = cv2.VideoCapture(idx, api, params) if params else cv2.VideoCapture(idx, api)
            except cv2.error:
                continue
            if not cap.isOpened():
                cap.release(); continue
            if name != "MSMF":
                if mjpg: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                if w and h: cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                if fps: cap.set(cv2.CAP_PROP_FPS, fps)
            if _works(cap):
                _last_good = name
                if not quiet: print(f"camera {idx}: opened with {name}")
                return cap, name
            cap.release()
    return None, None
