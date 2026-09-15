import threading
import unittest
from scripts.latest_frame_capture import LatestFrameCapture


class CaptureTests(unittest.TestCase):
    def test_frames_owned_and_ordered(self):
        counter = [0]
        def read():
            counter[0] += 1
            return [counter[0]]
        worker = LatestFrameCapture(read)
        worker.start()
        try:
            first = worker.next_after(0)
            second = worker.next_after(first.sequence)
            self.assertGreater(second.sequence, first.sequence)
            self.assertEqual(first.image, [first.sequence])
            self.assertEqual(second.image, [second.sequence])
        finally:
            worker.stop()
            worker.join(1)
        self.assertFalse(worker.is_alive())

    def test_exception_propagates(self):
        def read():
            raise ValueError('camera disconnected')
        worker = LatestFrameCapture(read)
        worker.start()
        try:
            with self.assertRaisesRegex(ValueError, 'disconnected'):
                worker.next_after(0)
        finally:
            worker.stop()
            worker.join(1)

    def test_stop_wakes_waiter(self):
        worker = LatestFrameCapture(lambda: None)
        worker.stop()
        with self.assertRaises(TimeoutError):
            worker.next_after(0, timeout=.01)

    def test_latest_slot_replaces_old_frames(self):
        worker = LatestFrameCapture(lambda: object())
        worker.start()
        try:
            item = worker.next_after(0)
            for _ in range(5):
                item = worker.next_after(item.sequence)
            self.assertEqual(worker.next_after(0).sequence, worker.latest.sequence)
            self.assertGreaterEqual(worker.statistics()['frames'], 6)
        finally:
            worker.stop()
            worker.join(1)


if __name__ == '__main__':
    unittest.main()
