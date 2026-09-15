"""FPGA-oriented entry point for the one-droplet hybrid pipeline.

The reference implementation optionally aligns consecutive patches with a
frequency-domain phase-correlation step. That operation is useful as a PC
diagnostic, but it is not a good first implementation for the Arty S7-25.

This entry point keeps the same CLI and output format while replacing that
alignment with a zero-copy temporal reference. Droplet-center filtering and
the particle tracker's spatial gate absorb the remaining small displacement.
"""

from __future__ import annotations

import run_microplastic_one_droplet_hybrid as pipeline


def use_previous_patch_without_phase_alignment(
    previous,
    current,
    hanning_window,
):
    del current, hanning_window
    return previous, (0.0, 0.0), 0.0


pipeline.align_previous_patch = use_previous_patch_without_phase_alignment


if __name__ == "__main__":
    pipeline.main()
