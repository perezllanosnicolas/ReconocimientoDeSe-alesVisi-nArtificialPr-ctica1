"""
Utility functions for geometric operations and Non-Maximum Suppression (NMS).
"""

from typing import List, Tuple


def calculate_iou(box_a: List[float], box_b: List[float]) -> Tuple[float, float]:
    """
    Calculates the Intersection over Union (IoU) and Intersection over Minimum (IoM)
    of two bounding boxes.

    Args:
        box_a: Bounding box A in format [x1, y1, x2, y2].
        box_b: Bounding box B in format [x1, y1, x2, y2].

    Returns:
        Tuple[float, float]: (IoU, IoM) metrics.
    """
    # Intersection coordinates
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Intersection area
    inter_area = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)

    # Areas of both individual boxes
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Calculate IoU (preventing division by zero)
    union_area = box_a_area + box_b_area - inter_area
    iou = inter_area / float(union_area) if union_area > 0 else 0.0

    # Calculate IoM (Intersection over Minimum, useful for fully enclosed boxes)
    min_area = min(box_a_area, box_b_area)
    iom = inter_area / float(min_area) if min_area > 0 else 0.0

    return iou, iom


def remove_overlapping_boxes(boxes: List[List[float]], iou_threshold: float = 0.2, iom_threshold: float = 0.8) -> List[List[float]]:
    """
    Removes redundant bounding boxes based on IoU and IoM overlap criteria 
    (Non-Maximum Suppression algorithm).
    
    Args:
        boxes: List of bounding boxes with format [x1, y1, x2, y2, score].
        iou_threshold: Maximum allowed Intersection over Union.
        iom_threshold: Maximum allowed Intersection over Minimum.
        
    Returns:
        List[List[float]]: Filtered list of non-overlapping bounding boxes.
    """
    if not boxes:
        return []

    # Sort boxes by score in descending order (highest confidence first)
    sorted_boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []

    for current_box in sorted_boxes:
        overlap = False
        for kept_box in keep:
            iou, iom = calculate_iou(current_box[:4], kept_box[:4])
            
            if iou > iou_threshold or iom > iom_threshold:
                overlap = True
                break
                
        if not overlap:
            keep.append(current_box)

    return keep
 