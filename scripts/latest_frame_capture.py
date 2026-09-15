"""One-owner acquisition thread with a bounded latest-frame slot."""
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    started_at: float
    completed_at: float
    image: object


class LatestFrameCapture(threading.Thread):
    def __init__(self, read):
        super().__init__(daemon=True)
        self.read = read
        self.condition = threading.Condition()
        self.stop_requested = threading.Event()
        self.latest = None
        self.error = None
        self.frames = 0
        self.total_read_seconds = 0.0
        self.first_started = None
        self.last_completed = None

    def run(self):
        try:
            while not self.stop_requested.is_set():
                started = time.perf_counter()
                image = self.read()
                completed = time.perf_counter()
                with self.condition:
                    self.frames += 1
                    self.total_read_seconds += completed - started
                    if self.first_started is None:
                        self.first_started = started
                    self.last_completed = completed
                    self.latest = CapturedFrame(self.frames, started, completed, image)
                    self.condition.notify_all()
                self.stop_requested.wait(0.001)
        except BaseException as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def next_after(self, sequence, timeout=5.0):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.error is not None or
                (self.latest is not None and self.latest.sequence > sequence) or
                self.stop_requested.is_set(), timeout,
            )
            if self.error is not None:
                raise self.error
            if not ready or self.latest is None or self.latest.sequence <= sequence:
                raise TimeoutError('No new camera frame')
            return self.latest

    def stop(self):
        self.stop_requested.set()
        with self.condition:
            self.condition.notify_all()

    def statistics(self):
        with self.condition:
            elapsed = (self.last_completed - self.first_started) if self.frames else 0
            return dict(frames=self.frames, stream_fps=self.frames/elapsed if elapsed else 0,
                        mean_read_ms=self.total_read_seconds*1000/self.frames if self.frames else None)
