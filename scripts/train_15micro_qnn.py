#!/usr/bin/env python3
"""Train TinyQuantDetector with anchors measured from the 15 um dataset."""

from __future__ import annotations

import qnn.train_qat as trainer
from qnn.model import DetectorConfig as BaseDetectorConfig


ANCHORS_15UM = (
    (0.03409375, 0.03471875),
    (0.21787500, 0.225109375),
)


def detector_config_15um(*args, **kwargs):
    kwargs["anchors"] = ANCHORS_15UM
    return BaseDetectorConfig(*args, **kwargs)


def main() -> None:
    trainer.DetectorConfig = detector_config_15um
    trainer.main()


if __name__ == "__main__":
    main()
