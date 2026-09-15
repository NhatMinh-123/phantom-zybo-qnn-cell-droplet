from __future__ import annotations

import unittest

import cv2
import numpy as np

from scripts.run_feature_ml_full_droplet_particle import (
    draw_particle_candidate,
    install_droplet_particle_geometry,
)


class FullDropletParticleOverlayTests(unittest.TestCase):
    def test_overlay_contains_full_droplet_box_and_particle_box(self) -> None:
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        original_circle, original_rectangle = (
            install_droplet_particle_geometry(128)
        )
        try:
            cv2.circle(image, (100, 100), 60, (0, 220, 255), 2)
            cv2.circle(image, (100, 100), 43, (255, 120, 0), 1)
            draw_particle_candidate(
                image,
                bbox=(108, 92, 4, 5),
                track_id=1,
                probability=0.9,
                hits=1,
                threshold=0.5,
                minimum_hits=3,
                label=False,
            )
        finally:
            cv2.circle = original_circle
            cv2.rectangle = original_rectangle

        self.assertTrue(np.any(image[40, 40] == (255, 120, 0)))
        self.assertTrue(np.any(image[92, 108] == (0, 180, 255)))
        self.assertTrue(np.all(image[100, 143] == 0))


if __name__ == "__main__":
    unittest.main()
