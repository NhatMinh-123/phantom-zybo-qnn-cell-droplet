from __future__ import annotations

import unittest

from scripts.dual_roi_temporal_counter import (
    ClassAwareLineTracker,
    CrossingEvent,
    CrossRoiAssociator,
    DetectionSample,
)


def detection(center_x: float, *, class_name: str = "droplet") -> DetectionSample:
    class_id = 1 if class_name == "droplet" else 0
    return DetectionSample(
        class_id,
        class_name,
        0.95,
        (center_x - 10.0, 40.0, center_x + 10.0, 70.0),
    )


class TemporalCounterTests(unittest.TestCase):
    def make_tracker(self, name: str = "verification") -> ClassAwareLineTracker:
        return ClassAwareLineTracker(
            region_name=name,
            roi_geometry=(100, 0, 220, 120),
            line_fraction=0.65,
            direction="left_to_right",
            minimum_hits={"cell": 3, "droplet": 2},
            max_misses={"cell": 3, "droplet": 4},
            max_center_distance={"cell": 0.30, "droplet": 0.40},
            count_hysteresis=0.03,
        )

    def test_counts_once_after_crossing(self) -> None:
        tracker = self.make_tracker()
        events = []
        for frame_index, center_x in enumerate((120.0, 145.0, 170.0, 190.0, 205.0)):
            events.extend(tracker.update([detection(center_x)], frame_index))
        self.assertEqual(len(events), 1)
        self.assertEqual(tracker.counts["droplet"], 1)

    def test_coasts_across_two_missing_updates_and_still_counts(self) -> None:
        tracker = self.make_tracker()
        events = []
        events.extend(tracker.update([detection(125.0)], 0))
        events.extend(tracker.update([detection(150.0)], 1))
        events.extend(tracker.update([], 2))
        events.extend(tracker.update([], 3))
        observations = tracker.observations(3)
        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0].predicted)
        events.extend(tracker.update([detection(190.0)], 4))
        self.assertEqual(len(events), 1)

    def test_counts_predicted_crossing_when_edge_detection_disappears(self) -> None:
        tracker = self.make_tracker()
        events = []
        events.extend(tracker.update([detection(120.0)], 0))
        events.extend(tracker.update([detection(150.0)], 1))
        events.extend(tracker.update([], 2))
        events.extend(tracker.update([], 3))
        events.extend(tracker.update([], 4))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].class_name, "droplet")
        self.assertEqual(tracker.counts["droplet"], 1)

    def test_does_not_count_track_born_after_line(self) -> None:
        tracker = self.make_tracker()
        events = []
        for frame_index, center_x in enumerate((185.0, 195.0, 205.0, 210.0)):
            events.extend(tracker.update([detection(center_x)], frame_index))
        self.assertEqual(events, [])

    def test_does_not_count_reverse_motion(self) -> None:
        tracker = self.make_tracker()
        events = []
        for frame_index, center_x in enumerate((205.0, 190.0, 170.0, 145.0, 120.0)):
            events.extend(tracker.update([detection(center_x)], frame_index))
        self.assertEqual(events, [])

    def test_associates_upstream_and_downstream_events(self) -> None:
        upstream = self.make_tracker("detection")
        downstream = self.make_tracker("verification")
        upstream_events = []
        downstream_events = []
        for frame_index, center_x in enumerate((120.0, 150.0, 185.0)):
            upstream_events.extend(upstream.update([detection(center_x)], frame_index))
        for frame_index, center_x in enumerate((120.0, 150.0, 185.0), start=10):
            downstream_events.extend(downstream.update([detection(center_x)], frame_index))
        associator = CrossRoiAssociator(
            minimum_delay_frames=1,
            maximum_delay_frames=30,
            maximum_cross_axis_distance=20.0,
            line_distance=192.0,
        )
        associator.add_upstream(upstream_events)
        match = associator.associate(downstream_events[0])
        self.assertTrue(match.verified)
        self.assertEqual(match.delay_frames, 10)

    def test_preserves_object_order_between_rois(self) -> None:
        def event(track_id: int, frame_index: int) -> CrossingEvent:
            return CrossingEvent(
                "candidate",
                track_id,
                1,
                "droplet",
                frame_index,
                180.0,
                55.0,
                0.95,
                3,
                5.0,
                0.0,
                track_id,
            )

        associator = CrossRoiAssociator(
            minimum_delay_frames=4,
            maximum_delay_frames=60,
            maximum_cross_axis_distance=20.0,
            line_distance=120.0,
            preserve_order=True,
        )
        associator.add_upstream([event(1, 10), event(2, 20)])
        match = associator.associate(event(10, 28))
        self.assertTrue(match.verified)
        self.assertEqual(match.upstream_event.track_id, 1)


if __name__ == "__main__":
    unittest.main()
