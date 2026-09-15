from tinygrid_qnn.config import TinyGridConfig
from tinygrid_qnn.labels import Box, boxes_to_occupancy, canvas_boxes_to_compact


def test_canvas_padding_is_removed_without_losing_edge_boxes() -> None:
    config = TinyGridConfig()
    boxes = [Box(1, 0.0, 48 / 384, 1.0, 336 / 384)]
    compact = canvas_boxes_to_compact(boxes, config)
    assert compact == [Box(1, 0.0, 0.0, 1.0, 1.0)]


def test_cell_and_droplet_can_occupy_the_same_grid_cell() -> None:
    config = TinyGridConfig()
    boxes = [
        Box(0, 0.49, 0.49, 0.51, 0.51),
        Box(1, 0.40, 0.40, 0.60, 0.60),
    ]
    target = boxes_to_occupancy(boxes, config)
    grid_x = int(0.5 * config.grid_width)
    grid_y = int(0.5 * config.grid_height)
    assert target[0, grid_y, grid_x] == 1
    assert target[1, grid_y, grid_x] == 1
