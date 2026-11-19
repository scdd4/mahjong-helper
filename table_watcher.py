"""Screenshot-only table change detection for optional auto-analysis.

No network interception, no game input, no memory reads. The watcher grabs the
same game window the manual button uses, reduces it to a small perceptual
signature focused on the hand row and centre river, and reports when the table
settles into a new state. The caller decides whether to run an analysis.
"""
import threading
import time


def signature(image):
    """Return a compact signature of the regions that change during play."""
    from PIL import ImageOps
    w, h = image.size
    hand = image.crop((0, int(h * .68), w, h))
    river = image.crop((int(w * .22), int(h * .14), int(w * .78), int(h * .72)))
    parts = []
    for region, size in ((hand, (48, 20)), (river, (32, 32))):
        gray = ImageOps.grayscale(region).resize(size)
        parts.extend(gray.tobytes())
    return parts


def difference(a, b):
    if not a or not b or len(a) != len(b):
        return 1.0
    return sum(abs(x - y) for x, y in zip(a, b)) / (len(a) * 255.0)


class TableWatcher:
    """Polls screenshots in a daemon thread; calls on_change when the table changes.

    A change must be seen in two consecutive polls before it fires, so transient
    animations do not trigger analysis. cooldown prevents repeated triggers from
    the same state.
    """

    def __init__(self, grab, on_change, interval=1.2, threshold=0.012, cooldown=8.0):
        self.grab = grab
        self.on_change = on_change
        self.interval = interval
        self.threshold = threshold
        self.cooldown = cooldown
        self._stop = threading.Event()
        self._thread = None
        self._last = None
        self._pending = None
        self._last_fired = 0.0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def reset(self):
        """Forget the current baseline (call after a manual capture/mode change)."""
        self._last = None
        self._pending = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                image = self.grab()
                if image is not None:
                    try:
                        current = signature(image)
                    finally:
                        image.close()
                    if self._last is None:
                        self._last = current
                    else:
                        delta = difference(self._last, current)
                        if delta >= self.threshold:
                            if self._pending is not None and difference(self._pending, current) < self.threshold * 0.5:
                                if time.monotonic() - self._last_fired >= self.cooldown:
                                    self._last_fired = time.monotonic()
                                    self._last = current
                                    self._pending = None
                                    self.on_change()
                            else:
                                self._pending = current
                        else:
                            self._pending = None
                            self._last = current
            except Exception:
                pass  # A failed poll must never stop watching.
            self._stop.wait(self.interval)
