import numpy as np

from tinygrid_qnn.preprocess import (
    bgr_to_gray_fixed,
    feature_maps_from_gray,
    resize_nearest_fixed,
    sobel_l1_fixed,
)


def test_gray_uses_documented_integer_coefficients() -> None:
    bgr = np.array([[[10, 20, 30], [255, 0, 0]]], dtype=np.uint8)
    expected = np.array(
        [[(77 * 30 + 150 * 20 + 29 * 10) >> 8, (29 * 255) >> 8]],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(bgr_to_gray_fixed(bgr), expected)


def test_nearest_resize_matches_integer_phase_mapping() -> None:
    source = np.arange(12, dtype=np.uint8).reshape(3, 4)
    resized = resize_nearest_fixed(source, 2, 2)
    np.testing.assert_array_equal(resized, np.array([[0, 2], [4, 6]], dtype=np.uint8))


def test_sobel_saturates_and_feature_order_is_fixed() -> None:
    current = np.zeros((5, 5), dtype=np.uint8)
    current[:, 3:] = 255
    previous = np.zeros_like(current)
    features = feature_maps_from_gray(current, previous)
    assert features.shape == (3, 5, 5)
    np.testing.assert_array_equal(features[0], current)
    np.testing.assert_array_equal(features[1], current)
    np.testing.assert_array_equal(features[2], sobel_l1_fixed(current))
    assert features[2].max() == 255
