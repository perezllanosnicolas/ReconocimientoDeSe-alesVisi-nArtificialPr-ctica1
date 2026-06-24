"""
Object Detection Evaluation Metrics Module.

Provides utilities to parse bounding box predictions, compute 
Intersection over Union (IoU), and generate Precision-Recall curves.
Allows benchmarking current model predictions against a baseline model.
"""

import cv2
import csv
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


class BoundingBox:
    """Class representing an object bounding box."""

    def __init__(self, left: float, top: float, right: float, bottom: float, 
                 class_id: int = -1, score: float = 1.0, img_idx: str = "-1"):
        self.left = int(float(left))
        self.top = int(float(top))
        self.right = int(float(right))
        self.bottom = int(float(bottom))
        self.class_id = int(class_id)
        self.score = float(score)
        self.img_idx = str(img_idx)

    def area(self) -> int:
        return (self.right - self.left + 1) * (self.bottom - self.top + 1)

    def opencv_plot(self, img: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255)) -> None:
        cv2.rectangle(img, (self.left, self.top), (self.right, self.bottom), color, thickness=2)
        cv2.putText(img, str(self.class_id), (self.left, int(self.top - 2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(img, str(self.score), (self.right, int(self.bottom + 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def __repr__(self) -> str:
        return str((self.img_idx, self.left, self.top, self.right, self.bottom, self.class_id, self.score))


def bboxes_overlap(gt_bbox: BoundingBox, dt_bbox: BoundingBox, ig: bool) -> float:
    """
    Computes the overlap area (Intersection over Union) of a ground truth (gt) 
    and detected (dt) bounding box.
    """
    w = min(dt_bbox.right, gt_bbox.right) - max(dt_bbox.left, gt_bbox.left)
    if w <= 0:
        return 0.0

    h = min(dt_bbox.bottom, gt_bbox.bottom) - max(dt_bbox.top, gt_bbox.top)
    if h <= 0:
        return 0.0
    
    intersection = w * h
    if ig:
        union = dt_bbox.area()
    else:
        union = dt_bbox.area() + gt_bbox.area() - intersection

    return intersection / union


def compute_class_index(number: int) -> int:
    """Maps specific traffic sign numerical IDs to broader category indices."""
    class_index = -1  # Default class index is -1 ("ignore")
    prohibitory_list = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
    mandatory_list = [38]
    danger_list = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    
    if number in prohibitory_list:
        class_index = 1
    elif number in danger_list:
        class_index = 2
    elif number == 14:
        class_index = 3  # stop
    elif number == 17:
        class_index = 4  # no-entry
    elif number == 13:
        class_index = 5  # yield
    elif number in mandatory_list:
        class_index = 6  # mandatory

    return class_index


def load_results_file(file_name: str, test_path: str, load_images: bool = False) -> Tuple[Dict, Dict]:
    """
    Parses a CSV/TXT detection results file.
    
    Returns:
        Tuple containing a dictionary of images and a dictionary of BoundingBox lists.
    """
    images = {}
    bboxes = {}
    
    if not os.path.exists(file_name):
        print(f"[WARNING] File not found: {file_name}")
        return images, bboxes

    with open(file_name, 'r') as gtfile:
        bbreader = csv.reader(gtfile, delimiter=';', quotechar='#')
        for row in bbreader:
            if not row:
                continue
                
            if load_images:
                image_path = os.path.join(test_path, row[0])
                if row[0] not in images:
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"[ERROR] Couldn't read image {image_path}")
                    else:
                        images[row[0]] = img

            if len(row) == 7:
                # Detections file (includes score)
                bb = BoundingBox(left=row[1], top=row[2], right=row[3], bottom=row[4],
                                 class_id=int(row[5]), score=float(row[6]), img_idx=str(row[0]))
            else:
                # Ground truth file (score defaults to 1.0)
                bb = BoundingBox(left=row[1], top=row[2], right=row[3], bottom=row[4],
                                 class_id=compute_class_index(int(row[5])),
                                 score=1.0, img_idx=str(row[0]))

            if row[0] not in bboxes:
                bboxes[row[0]] = []
            bboxes[row[0]].append(bb)
            
    return images, bboxes


def precision_recall_curve(gt_dbboxes: Dict, det_dbboxes: Dict, show: bool = False, 
                           ovr: float = 0.5, images_dict: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Computes the precision-recall curve arrays from detections and ground truth."""
    dimg = {}
    tot = 0
    for idx, bbox in sorted(gt_dbboxes.items(), key=lambda x: x[0]):
        if bbox:
            dimg[idx] = {"bbox": bbox, "det": [False] * len(gt_dbboxes)}
            for bbox_i in bbox:
                if bbox_i.class_id != -1:
                    tot += 1

    det_list = []
    for idx, bbox in sorted(det_dbboxes.items(), key=lambda x: x[0]):
        det_list.extend(bbox)

    det_list = sorted(det_list, reverse=True, key=lambda x: x.score)

    tp = np.zeros(len(det_list))
    fp = np.zeros(len(det_list))
    thr = np.zeros(len(det_list))
    
    for idx, det_bb in enumerate(det_list):
        found = False
        maxovr = 0.0
        gt = 0
        if det_bb.img_idx in dimg:
            bboxes = dimg[det_bb.img_idx]["bbox"]
            for ir, gt_bb in enumerate(bboxes):
                covr = bboxes_overlap(gt_bb, det_bb, ig=(gt_bb.class_id == -1))
                if covr >= maxovr:
                    maxovr = covr
                    gt = ir

        if maxovr > ovr:
            if dimg[det_bb.img_idx]["bbox"][gt].class_id != -1:
                if not (dimg[det_bb.img_idx]["det"][gt]):
                    tp[idx] = 1
                    dimg[det_bb.img_idx]["det"][gt] = True
                    found = True
                else:
                    fp[idx] = 1
        else:
            fp[idx] = 1

        thr[idx] = det_bb.score

    return tp, fp, thr, tot


def VOCap(rec: np.ndarray, prec: np.ndarray) -> float:
    """Computes AP based on PASCAL VOC metric."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, 0, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return float(ap)


def draw_PR_fast(tp: np.ndarray, fp: np.ndarray, tot: int, show: bool = True, col: str = "g") -> Tuple[np.ndarray, np.ndarray, float]:
    """Generates precision and recall arrays and computes average precision."""
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / tot if tot > 0 else np.zeros_like(tp_cum)
    prec = tp_cum / (fp_cum + tp_cum + 1e-9)
    ap1 = VOCap(rec, prec)
    return rec, prec, ap1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots Evaluation Results against a Baseline.')
    parser.add_argument('--test_path', required=True, help='Path to the testing dataset directory')
    parser.add_argument('--predictions_file', default="results/detections.txt", help='Results file of the current model')
    parser.add_argument('--baseline_file', default="data/baseline_detections.txt", help='Results file of the baseline model')
    args = parser.parse_args()

    print("[INFO] Loading current model predictions...")
    _, det_dbboxes = load_results_file(args.predictions_file, args.test_path)

    print("[INFO] Loading baseline predictions...")
    _, det_dbboxes_baseline = load_results_file(args.baseline_file, args.test_path)

    print("[INFO] Loading ground truth...")
    gt_file_path = os.path.join(args.test_path, "gt.txt")
    _, gt_dbboxes = load_results_file(gt_file_path, args.test_path)

    output_dir = "docs/assets"
    os.makedirs(output_dir, exist_ok=True)

    for iou_thresh in [0.5, 0.7]:
        print(f"\n[INFO] Computing metrics for IoU > {iou_thresh}...")
        
        # Current Model
        tp, fp, _, tot = precision_recall_curve(gt_dbboxes, det_dbboxes, show=False, ovr=iou_thresh)
        rec_det, prec_det, ap_det = draw_PR_fast(tp, fp, tot, show=False)

        # Baseline
        tp_b, fp_b, _, tot_b = precision_recall_curve(gt_dbboxes, det_dbboxes_baseline, show=False, ovr=iou_thresh)
        rec_b, prec_b, ap_b = draw_PR_fast(tp_b, fp_b, tot_b, show=False)

        plt.figure()
        plt.plot(rec_det, prec_det, '-r', label=f"Our Model (AP={ap_det * 100:.1f}%)")
        plt.plot(rec_b, prec_b, '-g', label=f"Baseline (AP={ap_b * 100:.1f}%)")
        plt.grid()
        plt.xlim((0, 1))
        plt.ylim((0, 1.1))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve (IoU > {iou_thresh})")
        plt.legend()
        
        save_path = os.path.join(output_dir, f"pr_curve_iou_{int(iou_thresh*100)}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[SUCCESS] Plot saved to {save_path}")