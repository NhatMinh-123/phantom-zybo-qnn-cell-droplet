"""Compare SDK Python and cached native reads without changing camera settings."""
import argparse
import json
import statistics
import time
from pathlib import Path
import sys
import cv2
from pyphantom import Phantom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_phantom_zybo_qnn_live import read_reduce8_bgr
from scripts.phantom_native_stream import NativePhantomReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--output', type=Path, default=Path('reports/ethernet_live_20260910/stream_optimization'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ph = Phantom()
    if ph.camera_count < 1:
        raise RuntimeError('No camera')
    cam = ph.Camera(0)
    results = []
    fast = None
    try:
        if int(cam.serial) != 25225:
            raise RuntimeError('Unexpected camera')
        # Initialize native reader after the wrapper establishes Reduce8 settings.
        read_reduce8_bgr(cam)
        native = NativePhantomReader(cam)
        modes = (
            ('python', None),
            ('native', None),
            ('fast', dict(demosaic_algorithm=1)),
            ('mono', dict(demosaic_algorithm=6)),
            ('crop_fast', dict(demosaic_algorithm=1, crop_rect=(645, 20, 816, 354))),
            ('crop_mono', dict(demosaic_algorithm=6, crop_rect=(645, 20, 816, 354))),
            ('python_repeat', None),
        )
        for name, settings in modes:
            if settings is not None:
                fast = NativePhantomReader(cam, **settings)
                read = fast.read
            else:
                read = native.read if name == 'native' else lambda: read_reduce8_bgr(cam)
            durations = []
            cpu_times = []
            for i in range(args.frames):
                start = time.perf_counter()
                cpu_start = time.process_time()
                frame = read()
                cpu_times.append(time.process_time()-cpu_start)
                durations.append(time.perf_counter()-start)
                if i == 0:
                    cv2.imwrite(str(args.output / f'{name}.png'), frame)
            item = dict(mode=name, frames=len(durations), fps=len(durations)/sum(durations),
                        mean_ms=statistics.mean(durations)*1000,
                        mean_process_cpu_ms=statistics.mean(cpu_times)*1000,
                        p95_ms=sorted(durations)[int(.95*(len(durations)-1))]*1000,
                        shape=list(frame.shape), durations_seconds=durations)
            results.append(item)
            print(json.dumps({k:v for k,v in item.items() if k != 'durations_seconds'}), flush=True)
            if fast is not None:
                fast.close()
                fast = None
        (args.output/'comparison.json').write_text(json.dumps(results,indent=2))
    finally:
        if fast is not None:
            fast.close()
        cam.close()
        ph.close()


if __name__ == '__main__':
    main()
