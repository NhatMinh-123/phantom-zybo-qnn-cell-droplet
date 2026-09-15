#!/usr/bin/env python3
"""Matplotlib 3.8 compatibility entry point for feature analysis."""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.axes import Axes


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.analyze_unlabeled_microplastic_features as analysis  # noqa: E402


ORIGINAL_BOXPLOT = Axes.boxplot


def compatible_boxplot(self, *args, **kwargs):
    if "tick_labels" in kwargs and "labels" not in kwargs:
        kwargs["labels"] = kwargs.pop("tick_labels")
    return ORIGINAL_BOXPLOT(self, *args, **kwargs)


def main() -> None:
    Axes.boxplot = compatible_boxplot
    analysis.main()


if __name__ == "__main__":
    main()
