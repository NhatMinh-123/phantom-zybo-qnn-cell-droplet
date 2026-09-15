import numpy as np
import torch

from qnn.dataset import translate_horizontal


def test_translate_horizontal_clips_and_drops_boxes() -> None:
    pixels = np.tile(np.arange(10, dtype=np.float32), (4, 1))
    labels = torch.tensor(
        [
            [1.0, 0.20, 0.50, 0.40, 0.20],
            [1.0, 0.10, 0.50, 0.10, 0.20],
        ]
    )

    translated, shifted = translate_horizontal(pixels, labels, -2)

    assert np.all(translated[:, -2:] == pixels[:, -1:])
    assert shifted.shape == (1, 5)
    assert torch.allclose(
        shifted[0], torch.tensor([1.0, 0.10, 0.50, 0.20, 0.20])
    )


def test_translate_horizontal_empty_labels() -> None:
    pixels = np.zeros((3, 8), dtype=np.float32)
    labels = torch.empty((0, 5), dtype=torch.float32)

    translated, shifted = translate_horizontal(pixels, labels, 2)

    assert translated.shape == pixels.shape
    assert shifted.shape == labels.shape
