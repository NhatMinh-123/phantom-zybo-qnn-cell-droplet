import unittest
from dataclasses import dataclass
from scripts.run_phantom_zybo_qnn_live import map_detections


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    box: tuple


class GeometryTests(unittest.TestCase):
    def test_upward_rotation_inverse(self):
        item = Detection(0, .9, (.2, .3, .6, .7))
        mapped = map_detections([item], ['droplet'], (100,200,200,300), 270)[0]
        self.assertEqual(mapped.box, (130,240,170,280))

    def test_downward_rotation_inverse(self):
        item = Detection(0, .9, (.2, .3, .6, .7))
        mapped = map_detections([item], ['droplet'], (100,200,200,300), 90)[0]
        self.assertEqual(mapped.box, (130,220,170,260))

    def test_original_geometry_unchanged(self):
        item = Detection(0, .9, (.2, .3, .6, .7))
        mapped = map_detections([item], ['droplet'], (100,200,200,300))[0]
        self.assertEqual(mapped.box, (120,230,160,270))


if __name__ == '__main__':
    unittest.main()
