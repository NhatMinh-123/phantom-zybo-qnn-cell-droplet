from __future__ import annotations

import unittest

import torch

from qnn.detection import decode_predictions, detector_loss, encode_targets
from qnn.model import DetectorConfig, TinyQuantDetector, config_from_dict


class DetectorV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DetectorConfig(
            input_width=192,
            input_height=144,
            downsample=8,
            slots_per_class=(2, 1),
            anchors=((0.049, 0.052), (0.347, 0.366)),
            weight_bits=8,
            activation_bits=8,
        )

    def test_rectangular_model_shape(self) -> None:
        output = TinyQuantDetector(self.config)(torch.rand(2, 1, 144, 192))
        self.assertEqual(tuple(output.shape), (2, 15, 18, 24))

    def test_two_cell_slots_remove_same_grid_collision(self) -> None:
        targets = [
            torch.tensor(
                [
                    [0, 0.500, 0.500, 0.050, 0.050],
                    [0, 0.510, 0.510, 0.040, 0.040],
                    [1, 0.500, 0.500, 0.350, 0.360],
                ]
            )
        ]
        encoded, collisions = encode_targets(
            targets,
            num_classes=2,
            grid_width=24,
            grid_height=18,
            slots_per_class=(2, 1),
            anchors=self.config.anchors,
            device=torch.device("cpu"),
        )
        self.assertEqual(collisions, 0)
        self.assertEqual(int(encoded[:, :, 0].sum().item()), 3)

    def test_loss_and_decode_accept_multiple_slots(self) -> None:
        predictions = torch.zeros((1, 15, 18, 24))
        targets = [torch.tensor([[0, 0.5, 0.5, 0.05, 0.05]])]
        loss = detector_loss(
            predictions,
            targets,
            num_classes=2,
            anchors=self.config.anchors,
            slots_per_class=self.config.slots_per_class,
        )
        detections = decode_predictions(
            predictions,
            confidence_threshold=0.9,
            anchors=self.config.anchors,
            slots_per_class=self.config.slots_per_class,
        )
        self.assertTrue(torch.isfinite(loss.total))
        self.assertEqual(detections, [[]])

    def test_aligned_iou_loss_penalizes_shifted_positive_box(self) -> None:
        target_x = 10.5 / 24
        target_y = 8.5 / 18
        targets = [
            torch.tensor(
                [[0, target_x, target_y, *self.config.anchors[0]]],
                dtype=torch.float32,
            )
        ]
        aligned = torch.zeros((1, 15, 18, 24), requires_grad=True)
        shifted = aligned.detach().clone()
        shifted[0, 1, 8, 10] = 2.0
        shifted.requires_grad_(True)

        def iou_component(predictions: torch.Tensor) -> torch.Tensor:
            baseline = detector_loss(
                predictions,
                targets,
                num_classes=2,
                anchors=self.config.anchors,
                slots_per_class=self.config.slots_per_class,
            )
            with_iou = detector_loss(
                predictions,
                targets,
                num_classes=2,
                anchors=self.config.anchors,
                slots_per_class=self.config.slots_per_class,
                iou_loss_weight=1.0,
            )
            return with_iou.box - baseline.box

        aligned_component = iou_component(aligned)
        shifted_component = iou_component(shifted)
        self.assertLess(float(aligned_component.detach()), 1e-4)
        self.assertGreater(float(shifted_component.detach()), 0.1)
        shifted_component.backward()
        self.assertTrue(torch.isfinite(shifted.grad).all())

    def test_legacy_configuration_defaults_to_one_slot(self) -> None:
        legacy = self.config.to_dict()
        legacy.pop("input_width")
        legacy.pop("input_height")
        legacy.pop("slots_per_class")
        legacy["input_size"] = 256
        restored = config_from_dict(legacy)
        self.assertEqual((restored.image_width, restored.image_height), (256, 256))
        self.assertEqual(restored.slots_per_class, (1, 1))

    def test_decode_accepts_class_specific_nms(self) -> None:
        predictions = torch.full((1, 15, 18, 24), -10.0)
        predictions[0, 10, 8, 10] = 8.0
        predictions[0, 10, 8, 11] = 7.0
        detections = decode_predictions(
            predictions,
            confidence_threshold=(0.9, 0.9),
            nms_iou=(0.45, 0.1),
            anchors=self.config.anchors,
            slots_per_class=self.config.slots_per_class,
        )
        droplets = [item for item in detections[0] if item.class_id == 1]
        self.assertEqual(len(droplets), 1)

    def test_decode_applies_class_box_constraints(self) -> None:
        predictions = torch.full((1, 15, 18, 24), -10.0)
        predictions[0, 10, 1, 10] = 8.0
        predictions[0, 10, 9, 10] = 7.0
        detections = decode_predictions(
            predictions,
            confidence_threshold=(0.9, 0.9),
            anchors=self.config.anchors,
            slots_per_class=self.config.slots_per_class,
            box_constraints=(None, {"center_y_min": 0.25, "center_y_max": 0.8}),
        )
        droplets = [item for item in detections[0] if item.class_id == 1]
        self.assertEqual(len(droplets), 1)
        center_y = (droplets[0].box[1] + droplets[0].box[3]) / 2
        self.assertGreaterEqual(center_y, 0.25)

    def test_decode_applies_post_nms_box_calibration(self) -> None:
        predictions = torch.full((1, 15, 18, 24), -10.0)
        predictions[0, 10:15, 9, 12] = 0.0
        predictions[0, 10, 9, 12] = 8.0
        baseline = decode_predictions(
            predictions,
            confidence_threshold=(0.9, 0.9),
            anchors=self.config.anchors,
            slots_per_class=self.config.slots_per_class,
        )[0][0]
        calibrated = decode_predictions(
            predictions,
            confidence_threshold=(0.9, 0.9),
            anchors=self.config.anchors,
            slots_per_class=self.config.slots_per_class,
            box_calibration=(
                None,
                {
                    "width_scale": 1.075,
                    "height_scale": 1.275,
                    "center_x_offset": 0.0,
                    "center_y_offset": 0.0,
                },
            ),
        )[0][0]
        baseline_width = baseline.box[2] - baseline.box[0]
        baseline_height = baseline.box[3] - baseline.box[1]
        calibrated_width = calibrated.box[2] - calibrated.box[0]
        calibrated_height = calibrated.box[3] - calibrated.box[1]
        self.assertAlmostEqual(calibrated_width / baseline_width, 1.075, places=5)
        self.assertAlmostEqual(calibrated_height / baseline_height, 1.275, places=5)


if __name__ == "__main__":
    unittest.main()
