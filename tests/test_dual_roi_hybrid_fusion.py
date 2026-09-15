import unittest

from scripts.dual_roi_hybrid_fusion import CrossBranchConsensus, HybridFusionConfig
from scripts.dual_roi_temporal_counter import CrossingEvent


def event(region, track, cls, frame, x=50.0, y=50.0, confidence=0.9):
    return CrossingEvent(
        region_name=region,
        track_id=track,
        class_id=0 if cls == "cell" else 1,
        class_name=cls,
        frame_index=frame,
        center_x=x,
        center_y=y,
        confidence=confidence,
        hits=3,
        velocity_x=5.0,
        velocity_y=0.0,
        cumulative_count=1,
    )


class CrossBranchConsensusTest(unittest.TestCase):
    def make_fusion(self):
        return CrossBranchConsensus(
            HybridFusionConfig(
                minimum_delay_frames=2,
                maximum_delay_frames=20,
                maximum_cross_axis_distance=15.0,
                nominal_delay_frames=8.0,
            )
        )

    def test_requires_qnn_and_classical_same_class(self):
        fusion = self.make_fusion()
        self.assertEqual([], fusion.add_qnn([event("qnn", 1, "droplet", 10)], 10))
        output = fusion.add_classical([event("classical", 4, "droplet", 18)], 18)
        self.assertEqual(1, len(output))
        self.assertEqual("droplet", output[0].class_name)
        self.assertEqual(1, fusion.counts["droplet"])

    def test_rejects_class_mismatch(self):
        fusion = self.make_fusion()
        fusion.add_qnn([event("qnn", 1, "cell", 10)], 10)
        output = fusion.add_classical([event("classical", 4, "droplet", 18)], 18)
        self.assertEqual([], output)
        self.assertEqual({}, fusion.counts)

    def test_rejects_downstream_event_before_upstream(self):
        fusion = self.make_fusion()
        fusion.add_classical([event("classical", 4, "droplet", 8)], 8)
        output = fusion.add_qnn([event("qnn", 1, "droplet", 10)], 10)
        self.assertEqual([], output)

    def test_handles_host_result_arriving_after_downstream_event(self):
        fusion = self.make_fusion()
        fusion.add_classical([event("classical", 4, "cell", 18)], 18)
        output = fusion.add_qnn([event("qnn", 1, "cell", 10)], 19)
        self.assertEqual(1, len(output))
        self.assertEqual(8, output[0].delay_frames)

    def test_one_to_one_consumption(self):
        fusion = self.make_fusion()
        fusion.add_qnn(
            [event("qnn", 1, "droplet", 10), event("qnn", 2, "droplet", 12)],
            12,
        )
        first = fusion.add_classical([event("classical", 5, "droplet", 18)], 18)
        second = fusion.add_classical([event("classical", 6, "droplet", 20)], 20)
        self.assertEqual(1, len(first))
        self.assertEqual(1, len(second))
        self.assertEqual(2, fusion.counts["droplet"])


if __name__ == "__main__":
    unittest.main()

