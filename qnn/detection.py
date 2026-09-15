from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DetectionLoss:
    total: torch.Tensor
    objectness: torch.Tensor
    box: torch.Tensor
    positives: int
    collisions: int


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]


def encode_targets(
    targets: list[torch.Tensor],
    *,
    num_classes: int,
    grid_size: int | None = None,
    grid_width: int | None = None,
    grid_height: int | None = None,
    slots_per_class: tuple[int, ...] | None = None,
    anchors: tuple[tuple[float, float], ...],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    if grid_size is not None:
        grid_width = grid_height = grid_size
    if grid_width is None or grid_height is None:
        raise ValueError("grid_size or both grid_width/grid_height are required")
    slots_per_class = slots_per_class or (1,) * num_classes
    if len(slots_per_class) != num_classes:
        raise ValueError("slots_per_class must have one value per class")
    class_slot_offsets = []
    slot_offset = 0
    for slot_count in slots_per_class:
        class_slot_offsets.append(slot_offset)
        slot_offset += slot_count
    encoded = torch.zeros(
        (len(targets), slot_offset, 5, grid_height, grid_width),
        dtype=torch.float32,
        device=device,
    )
    collisions = 0
    for batch_index, boxes in enumerate(targets):
        for row in boxes.to(device):
            class_id = int(row[0].item())
            center_x, center_y, width, height = row[1:]
            scaled_x = center_x * grid_width
            scaled_y = center_y * grid_height
            grid_x = min(int(scaled_x.item()), grid_width - 1)
            grid_y = min(int(scaled_y.item()), grid_height - 1)
            first_slot = class_slot_offsets[class_id]
            class_slots = range(first_slot, first_slot + slots_per_class[class_id])
            free_slot = next(
                (
                    slot_id
                    for slot_id in class_slots
                    if encoded[batch_index, slot_id, 0, grid_y, grid_x] <= 0
                ),
                None,
            )
            slot_id = free_slot
            if slot_id is None:
                collisions += 1
                anchor_width, anchor_height = anchors[class_id]
                areas = []
                for candidate in class_slots:
                    old_width = anchor_width * torch.exp(
                        encoded[batch_index, candidate, 3, grid_y, grid_x]
                    )
                    old_height = anchor_height * torch.exp(
                        encoded[batch_index, candidate, 4, grid_y, grid_x]
                    )
                    areas.append(old_width * old_height)
                smallest_index = int(torch.stack(areas).argmin().item())
                slot_id = first_slot + smallest_index
                if width * height <= areas[smallest_index]:
                    continue
            encoded[batch_index, slot_id, 0, grid_y, grid_x] = 1.0
            encoded[batch_index, slot_id, 1, grid_y, grid_x] = scaled_x - grid_x
            encoded[batch_index, slot_id, 2, grid_y, grid_x] = scaled_y - grid_y
            anchor_width, anchor_height = anchors[class_id]
            encoded[batch_index, slot_id, 3, grid_y, grid_x] = torch.log(
                width / anchor_width
            )
            encoded[batch_index, slot_id, 4, grid_y, grid_x] = torch.log(
                height / anchor_height
            )
    return encoded, collisions


def detector_loss(
    predictions: torch.Tensor,
    targets: list[torch.Tensor],
    *,
    num_classes: int,
    anchors: tuple[tuple[float, float], ...],
    slots_per_class: tuple[int, ...] | None = None,
    box_weight: float = 5.0,
    focal_gamma: float = 2.0,
    iou_loss_weight: float = 0.0,
) -> DetectionLoss:
    batch_size, channels, grid_h, grid_w = predictions.shape
    slots_per_class = slots_per_class or (1,) * num_classes
    total_slots = sum(slots_per_class)
    if channels != total_slots * 5:
        raise ValueError("Unexpected detector output shape")
    predictions = predictions.view(batch_size, total_slots, 5, grid_h, grid_w)
    encoded, collisions = encode_targets(
        targets,
        num_classes=num_classes,
        grid_width=grid_w,
        grid_height=grid_h,
        slots_per_class=slots_per_class,
        anchors=anchors,
        device=predictions.device,
    )

    object_logits = predictions[:, :, 0]
    object_targets = encoded[:, :, 0]
    positive_mask = object_targets > 0.5
    negative_mask = ~positive_mask
    probabilities = torch.sigmoid(object_logits)
    positive_bce = F.softplus(-object_logits[positive_mask])
    negative_bce = F.softplus(object_logits[negative_mask])
    positive_loss = (
        ((1.0 - probabilities[positive_mask]).pow(focal_gamma) * positive_bce).mean()
        if positive_mask.any()
        else object_logits.sum() * 0.0
    )
    negative_loss = (
        (probabilities[negative_mask].pow(focal_gamma) * negative_bce).mean()
        if negative_mask.any()
        else object_logits.sum() * 0.0
    )
    # Positive and negative cells are normalized separately. This prevents a
    # sparse 64x64 grid from converging to the all-background solution.
    objectness_loss = 0.75 * positive_loss + 0.25 * negative_loss

    positive_count = int(positive_mask.sum().item())
    if positive_count:
        offset_mask = positive_mask.unsqueeze(2).expand(-1, -1, 2, -1, -1)
        size_mask = positive_mask.unsqueeze(2).expand(-1, -1, 2, -1, -1)
        offset_loss = F.smooth_l1_loss(
            torch.sigmoid(predictions[:, :, 1:3])[offset_mask],
            encoded[:, :, 1:3][offset_mask],
            reduction="mean",
        )
        size_loss = F.smooth_l1_loss(
            predictions[:, :, 3:5][size_mask],
            encoded[:, :, 3:5][size_mask],
            reduction="mean",
        )
        box_loss = offset_loss + 0.5 * size_loss
        if iou_loss_weight > 0.0:
            positive_indices = positive_mask.nonzero(as_tuple=False)
            batch_index = positive_indices[:, 0]
            slot_index = positive_indices[:, 1]
            grid_y = positive_indices[:, 2]
            grid_x = positive_indices[:, 3]
            positive_predictions = predictions[
                batch_index, slot_index, :, grid_y, grid_x
            ]
            positive_targets = encoded[
                batch_index, slot_index, :, grid_y, grid_x
            ]
            slot_class_ids = torch.tensor(
                [
                    class_id
                    for class_id, slot_count in enumerate(slots_per_class)
                    for _ in range(slot_count)
                ],
                dtype=torch.long,
                device=predictions.device,
            )
            anchor_values = torch.tensor(
                anchors,
                dtype=predictions.dtype,
                device=predictions.device,
            )[slot_class_ids[slot_index]]
            predicted_center = torch.stack(
                (
                    (
                        grid_x.to(predictions.dtype)
                        + torch.sigmoid(positive_predictions[:, 1])
                    )
                    / grid_w,
                    (
                        grid_y.to(predictions.dtype)
                        + torch.sigmoid(positive_predictions[:, 2])
                    )
                    / grid_h,
                ),
                dim=1,
            )
            target_center = torch.stack(
                (
                    (
                        grid_x.to(predictions.dtype)
                        + positive_targets[:, 1]
                    )
                    / grid_w,
                    (
                        grid_y.to(predictions.dtype)
                        + positive_targets[:, 2]
                    )
                    / grid_h,
                ),
                dim=1,
            )
            predicted_size = anchor_values * torch.exp(
                positive_predictions[:, 3:5].clamp(-3.0, 2.0)
            )
            target_size = anchor_values * torch.exp(positive_targets[:, 3:5])
            predicted_box = torch.cat(
                (
                    predicted_center - predicted_size / 2,
                    predicted_center + predicted_size / 2,
                ),
                dim=1,
            )
            target_box = torch.cat(
                (
                    target_center - target_size / 2,
                    target_center + target_size / 2,
                ),
                dim=1,
            )
            top_left = torch.maximum(predicted_box[:, :2], target_box[:, :2])
            bottom_right = torch.minimum(predicted_box[:, 2:], target_box[:, 2:])
            intersection_size = (bottom_right - top_left).clamp(min=0.0)
            intersection = intersection_size[:, 0] * intersection_size[:, 1]
            predicted_area = predicted_size[:, 0] * predicted_size[:, 1]
            target_area = target_size[:, 0] * target_size[:, 1]
            aligned_iou = intersection / (
                predicted_area + target_area - intersection + 1e-7
            )
            box_loss = box_loss + iou_loss_weight * (1.0 - aligned_iou).mean()
    else:
        box_loss = predictions.sum() * 0.0

    total = objectness_loss + box_weight * box_loss
    return DetectionLoss(total, objectness_loss, box_loss, positive_count, collisions)


def box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    first_area = (first[:, 2] - first[:, 0]).clamp(min=0) * (
        first[:, 3] - first[:, 1]
    ).clamp(min=0)
    second_area = (second[:, 2] - second[:, 0]).clamp(min=0) * (
        second[:, 3] - second[:, 1]
    ).clamp(min=0)
    return intersection_area / (first_area[:, None] + second_area[None, :] - intersection_area + 1e-7)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> list[int]:
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel():
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        overlaps = box_iou(boxes[current].unsqueeze(0), boxes[remaining]).squeeze(0)
        order = remaining[overlaps <= iou_threshold]
    return keep


def decode_predictions(
    predictions: torch.Tensor,
    *,
    confidence_threshold: float | tuple[float, ...] = 0.35,
    nms_iou: float | tuple[float, ...] = 0.45,
    pre_nms_topk: int = 100,
    max_detections: int = 200,
    anchors: tuple[tuple[float, float], ...] = ((0.030, 0.031), (0.210, 0.219)),
    slots_per_class: tuple[int, ...] | None = None,
    box_constraints: tuple[dict[str, float] | None, ...] | None = None,
    box_calibration: tuple[dict[str, float] | None, ...] | None = None,
) -> list[list[Detection]]:
    batch_size, channels, grid_h, grid_w = predictions.shape
    num_classes = len(anchors)
    slots_per_class = slots_per_class or (1,) * num_classes
    total_slots = sum(slots_per_class)
    if channels != total_slots * 5:
        raise ValueError("Unexpected detector output shape")
    if isinstance(confidence_threshold, tuple):
        if len(confidence_threshold) != num_classes:
            raise ValueError("confidence_threshold must have one value per class")
        class_thresholds = confidence_threshold
    else:
        class_thresholds = (confidence_threshold,) * num_classes
    if isinstance(nms_iou, tuple):
        if len(nms_iou) != num_classes:
            raise ValueError("nms_iou must have one value per class")
        class_nms_iou = nms_iou
    else:
        class_nms_iou = (nms_iou,) * num_classes
    if box_constraints is None:
        box_constraints = (None,) * num_classes
    if len(box_constraints) != num_classes:
        raise ValueError("box_constraints must have one value per class")
    if box_calibration is None:
        box_calibration = (None,) * num_classes
    if len(box_calibration) != num_classes:
        raise ValueError("box_calibration must have one value per class")
    values = predictions.view(batch_size, total_slots, 5, grid_h, grid_w)
    slot_class_ids = tuple(
        class_id
        for class_id, slot_count in enumerate(slots_per_class)
        for _ in range(slot_count)
    )
    output: list[list[Detection]] = []

    for batch_index in range(batch_size):
        image_detections: list[Detection] = []
        for class_id in range(num_classes):
            class_slots = [
                slot_id for slot_id, owner in enumerate(slot_class_ids) if owner == class_id
            ]
            confidence = torch.sigmoid(values[batch_index, class_slots, 0])
            local_maximum = F.max_pool2d(
                confidence[:, None], kernel_size=3, stride=1, padding=1
            )[:, 0]
            locations = torch.nonzero(
                (confidence >= class_thresholds[class_id])
                & (confidence >= local_maximum - 1e-7),
                as_tuple=False,
            )
            if not len(locations):
                continue
            local_slot, grid_y, grid_x = locations[:, 0], locations[:, 1], locations[:, 2]
            slot_ids = torch.tensor(class_slots, device=predictions.device)[local_slot]
            scores = confidence[local_slot, grid_y, grid_x]
            if len(scores) > pre_nms_topk:
                _, top_indices = scores.topk(pre_nms_topk)
                grid_y = grid_y[top_indices]
                grid_x = grid_x[top_indices]
                slot_ids = slot_ids[top_indices]
                scores = scores[top_indices]
            raw_boxes = values[batch_index, slot_ids, 1:5, grid_y, grid_x]
            center_x = (grid_x + torch.sigmoid(raw_boxes[:, 0])) / grid_w
            center_y = (grid_y + torch.sigmoid(raw_boxes[:, 1])) / grid_h
            anchor_width, anchor_height = anchors[class_id]
            width = anchor_width * torch.exp(raw_boxes[:, 2].clamp(-3.0, 2.0))
            height = anchor_height * torch.exp(raw_boxes[:, 3].clamp(-3.0, 2.0))
            boxes = torch.stack(
                (
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                ),
                dim=1,
            ).clamp(0.0, 1.0)
            constraints = box_constraints[class_id]
            if constraints:
                box_width = boxes[:, 2] - boxes[:, 0]
                box_height = boxes[:, 3] - boxes[:, 1]
                box_center_x = (boxes[:, 0] + boxes[:, 2]) / 2
                box_center_y = (boxes[:, 1] + boxes[:, 3]) / 2
                valid = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
                values_by_name = {
                    "center_x": box_center_x,
                    "center_y": box_center_y,
                    "width": box_width,
                    "height": box_height,
                }
                for name, values_for_box in values_by_name.items():
                    minimum = constraints.get(f"{name}_min")
                    maximum = constraints.get(f"{name}_max")
                    if minimum is not None:
                        valid &= values_for_box >= minimum
                    if maximum is not None:
                        valid &= values_for_box <= maximum
                boxes = boxes[valid]
                scores = scores[valid]
                if not len(scores):
                    continue
            for index in _nms(boxes, scores, class_nms_iou[class_id])[:max_detections]:
                selected_box = boxes[index]
                calibration = box_calibration[class_id]
                if calibration:
                    box_center_x = (selected_box[0] + selected_box[2]) / 2
                    box_center_y = (selected_box[1] + selected_box[3]) / 2
                    box_width = (selected_box[2] - selected_box[0]) * float(
                        calibration.get("width_scale", 1.0)
                    )
                    box_height = (selected_box[3] - selected_box[1]) * float(
                        calibration.get("height_scale", 1.0)
                    )
                    box_center_x += float(calibration.get("center_x_offset", 0.0))
                    box_center_y += float(calibration.get("center_y_offset", 0.0))
                    selected_box = torch.stack(
                        (
                            box_center_x - box_width / 2,
                            box_center_y - box_height / 2,
                            box_center_x + box_width / 2,
                            box_center_y + box_height / 2,
                        )
                    ).clamp(0.0, 1.0)
                image_detections.append(
                    Detection(
                        class_id=class_id,
                        confidence=float(scores[index].item()),
                        box=tuple(float(value) for value in selected_box.tolist()),
                    )
                )
        image_detections.sort(key=lambda item: item.confidence, reverse=True)
        output.append(image_detections[:max_detections])
    return output


def count_matches(
    detections: list[list[Detection]],
    targets: list[torch.Tensor],
    *,
    num_classes: int,
    iou_threshold: float = 0.5,
) -> tuple[list[int], list[int], list[int]]:
    true_positive = [0] * num_classes
    false_positive = [0] * num_classes
    false_negative = [0] * num_classes

    for image_detections, image_targets in zip(detections, targets):
        for class_id in range(num_classes):
            class_targets = image_targets[image_targets[:, 0] == class_id]
            if len(class_targets):
                centers = class_targets[:, 1:3]
                sizes = class_targets[:, 3:5]
                target_boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=1)
            else:
                target_boxes = torch.empty((0, 4))
            class_detections = [item for item in image_detections if item.class_id == class_id]
            matched: set[int] = set()
            for detection in class_detections:
                if not len(target_boxes):
                    false_positive[class_id] += 1
                    continue
                detection_box = torch.tensor(detection.box).view(1, 4)
                overlaps = box_iou(detection_box, target_boxes).squeeze(0)
                best_iou, best_index = overlaps.max(dim=0)
                target_index = int(best_index.item())
                if best_iou >= iou_threshold and target_index not in matched:
                    true_positive[class_id] += 1
                    matched.add(target_index)
                else:
                    false_positive[class_id] += 1
            false_negative[class_id] += len(target_boxes) - len(matched)
    return true_positive, false_positive, false_negative
