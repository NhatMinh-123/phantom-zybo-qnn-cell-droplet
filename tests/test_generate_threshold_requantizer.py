from pathlib import Path

import numpy as np

from finn.generate_threshold_requantizer import (
    parse_threshold_header,
    quantize_accumulators,
    render_vhdl,
)


def test_parse_and_quantize_thresholds(tmp_path: Path) -> None:
    header = tmp_path / "thresh.h"
    rows = [list(range(-127, 128)), list(range(-126, 129))]
    values = ", ".join(str(value) for row in rows for value in row)
    header.write_text(
        "static ThresholdsActivation<2,1,255,ap_int<18>,ap_int<8>,"
        f"-128> threshs = {{{{{values}}}}};\n",
        encoding="ascii",
    )

    thresholds = parse_threshold_header(header)
    accumulators = np.asarray([[[-128, -128], [0, 0], [127, 128]]])
    result = quantize_accumulators(accumulators, thresholds)

    assert thresholds.shape == (2, 255)
    assert result.tolist() == [[[-128, -128], [0, -1], [127, 127]]]
    rendered = render_vhdl(thresholds)
    assert "DETECTOR_OUTPUT_CHANNELS : positive := 2" in rendered
    assert "509 => to_signed(128, DETECTOR_THRESHOLD_BITS)" in rendered
