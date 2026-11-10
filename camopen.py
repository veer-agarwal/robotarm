"""Open a camera on Windows without guessing the backend.

DirectShow is usually the best choice, but on some PCs it can't open anything ("can't be used to capture by index")
while Media Foundation (MSMF) works. open_capture tries them in order and keeps the first that delivers a frame.
"""
import os
import cv2

ORDER = ["DSHOW", "MSMF", "ANY"] if os.name == "nt" else ["ANY"]
_last_good = None


def open_capture(idx, prefer=None, quiet=False):
    """Return (cap, backend_name), or (None, None). prefer = 'DSHOW' / 'MSMF' / 'ANY' to try that one first."""
    global _last_good
    order = list(ORDER)
    for first in (prefer, _last_good):
        if first and first in order:
            order.remove(first); order.insert(0, first)
    for name in order:
        api = getattr(cv2, f"CAP_{name}")
        cap = cv2.VideoCapture(idx, api)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                _last_good = name
                if not quiet: print(f"camera {idx}: opened with {name}")
                return cap, name
        cap.release()
    return None, None
