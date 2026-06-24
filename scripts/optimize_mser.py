"""
Hyperparameter Optimization Script (Grid Search) for the MSER Detector.

This module performs an exhaustive grid search over the hyperparameter space
to maximize the geometric F1-Score against the official Ground Truth,
ensuring an objective calibration of the MSER detection algorithm.
"""

import argparse
import itertools
import os
import glob
import cv2
from typing import Dict, List, Tuple, Set

from src.detector_mser import PanelDetectorMSER
from src.utils import calculate_iou, remove_overlapping_boxes


def load_ground_truth(txt_path: str) -> Dict[str, List[List[int]]]:
    """
    Loads and parses the Ground Truth annotations file.

    Args:
        txt_path (str): Path to the text file containing the real bounding boxes.

    Returns:
        Dict[str, List[List[int]]]: Structured dictionary mapping image names 
                                    to a list of bounding boxes [x1, y1, x2, y2].
    """
    results = {}
    if not os.path.exists(txt_path):
        print(f"[ERROR] Ground Truth file not found at: {txt_path}")
        return results
        
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            # Format expected: image.png;x1;y1;x2;y2;class;score
            if len(parts) >= 5:
                img_name = parts[0]
                box = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]
                if img_name not in results:
                    results[img_name] = []
                results[img_name].append(box)
    return results


def evaluate_configuration(detector, test_path: str, ground_truth: Dict, iou_threshold: float = 0.5) -> Tuple[float, float, float]:
    """
    Evaluates a specific detector configuration purely in memory (no disk I/O).
    Uses the Pascal VOC strict criterion (IoU > 0.5) to determine True Positives.
    """
    test_images = glob.glob(os.path.join(test_path, "*.png"))
    if not test_images:
        return 0.0, 0.0, 0.0
    
    true_positives = 0
    false_positives = 0
    total_gt_boxes = sum(len(boxes) for boxes in ground_truth.values())

    for img_path in test_images:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: 
            continue
            
        # 1. Inference phase
        raw_boxes = detector.detect(img)
        
        # 2. Non-Maximum Suppression (NMS) equivalent
        clean_boxes = remove_overlapping_boxes(raw_boxes, iou_threshold=0.2, iom_threshold=0.8)
        our_boxes = [box[:4] for box in clean_boxes]
        
        # 3. Evaluation phase against Ground Truth
        gt_boxes_img = ground_truth.get(img_name, [])
        matched_gt_indices: Set[int] = set()

        for our_box in our_boxes:
            match_found = False
            for gt_idx, gt_box in enumerate(gt_boxes_img):
                if gt_idx in matched_gt_indices: 
                    continue
                
                iou, _ = calculate_iou(our_box, gt_box)
                
                if iou > iou_threshold:
                    true_positives += 1
                    matched_gt_indices.add(gt_idx)
                    match_found = True
                    break
            
            if not match_found:
                false_positives += 1

    # Final metrics calculation
    recall = true_positives / total_gt_boxes if total_gt_boxes > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1_score, precision, recall


def main() -> None:
    parser = argparse.ArgumentParser(description="MSER Detector Hyperparameter Optimizer")
    parser.add_argument('--test_path', required=True, help="Path to the test images directory")
    parser.add_argument('--gt_file', required=True, help="Path to the Ground Truth text file")
    args = parser.parse_args()

    print("=" * 60)
    print("STARTING HYPERPARAMETER OPTIMIZATION (MSER DETECTOR)")
    print("=" * 60)
    
    ground_truth = load_ground_truth(args.gt_file)
    if not ground_truth: 
        print("[ERROR] Could not load Ground Truth. Aborting.")
        return

    # Search Space Definition
    grid_parameters = {
        'mser_min_area': [500],
        'hsv_hue_min': [90],
        'hsv_hue_max': [130],
        'hsv_sat_min': [150],
        'score_threshold': [0.2],
        'delta': [2],
        'mask_height': [40],
        'mask_width': [80],
        'min_box_area': [1000],
        'min_box_width': [18],
        'min_box_height': [18],
        'min_aspect_ratio': [0.6],
        'max_aspect_ratio': [4]
    }
    
    param_names = list(grid_parameters.keys())
    param_values = list(grid_parameters.values())
    combinations = list(itertools.product(*param_values))
    
    best_f1 = -1.0
    best_params_dict = {}
    
    total = len(combinations)
    print(f"[*] Generated search space: {total} combinations.")
    print("[*] Evaluating configurations...\n")

    for i, combination in enumerate(combinations):
        params = dict(zip(param_names, combination))
        
        detector = PanelDetectorMSER(
            min_area=params['mser_min_area'],
            hue_min=params['hsv_hue_min'],
            hue_max=params['hsv_hue_max'],
            sat_min=params['hsv_sat_min'],
            score_threshold=params['score_threshold'],
            delta=params['delta'],
            mask_height=params['mask_height'],
            mask_width=params['mask_width'],
            min_box_area=params['min_box_area'],
            min_box_width=params['min_box_width'],
            min_box_height=params['min_box_height'],
            min_aspect_ratio=params['min_aspect_ratio'],
            max_aspect_ratio=params['max_aspect_ratio']
        )
        
        f1, prec, rec = evaluate_configuration(detector, args.test_path, ground_truth)
        
        if f1 > best_f1:
            best_f1 = f1
            best_params_dict = params
            print(f" [Iter {i+1:06d}/{total}]  NEW GLOBAL OPTIMUM")
            print(f"   -> F1-Score: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
            print(f"   -> Parameters: {params}\n")
        elif (i + 1) % 10 == 0:
            print(f"   [Progress] {i+1}/{total} combinations analyzed...")

    print("\n" + "=" * 60)
    print(" OPTIMIZATION COMPLETED SUCCESSFULLY ")
    print(f" Best F1-Score: {best_f1:.4f}")
    print("\n WINNING CONFIGURATION:")
    for k, v in best_params_dict.items():
        print(f"  - {k} = {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()