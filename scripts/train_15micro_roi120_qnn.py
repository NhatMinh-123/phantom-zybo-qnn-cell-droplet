#!/usr/bin/env python3
"""Train the ROI120 TinyQuantDetector with anchors measured after cropping."""

from __future__ import annotations

import qnn.train_qat as trainer
from qnn.model import DetectorConfig as BaseDetectorConfig


ANCHORS_15UM_ROI120 = (
    (0.07406667, 0.07446667),
    (0.42776667, 0.49260000),
)


def detector_config_15um_roi120(*args, **kwargs):
    kwargs["anchors"] = ANCHORS_15UM_ROI120
    return BaseDetectorConfig(*args, **kwargs)


def main() -> None:
    trainer.DetectorConfig = detector_config_15um_roi120
    trainer.main()


if __name__ == "__main__":
    main()
