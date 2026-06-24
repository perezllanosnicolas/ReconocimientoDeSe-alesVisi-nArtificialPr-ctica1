"""
Average Precision (AP) Evaluation Script.

Calculates the Average Precision metric at IoU thresholds of 0.5 and 0.7
for a given set of object detection predictions against ground truth.
"""

import argparse
import os
from scripts.evaluar_resultados import load_results_file, precision_recall_curve, draw_PR_fast


def compute_ap(predictions_file: str, dataset_path: str, iou: float) -> float:
    """
    Computes Average Precision for a specific IoU threshold.

    Args:
        predictions_file (str): Path to the txt file containing model predictions.
        dataset_path (str): Path to the dataset directory containing the 'gt.txt' file.
        iou (float): Intersection over Union threshold.

    Returns:
        float: Computed Average Precision (AP) score.
    """
    gt_file_path = os.path.join(dataset_path, "gt.txt")
    
    _, det_dbboxes = load_results_file(predictions_file, dataset_path)
    _, gt_dbboxes = load_results_file(gt_file_path, dataset_path)

    tp, fp, _, tot = precision_recall_curve(
        gt_dbboxes,
        det_dbboxes,
        show=False,
        ovr=iou,
    )
    
    _, _, ap = draw_PR_fast(tp, fp, tot, show=False)
    return float(ap)


def main() -> None:
    """Main execution function to compute and print AP metrics."""
    parser = argparse.ArgumentParser(
        description="Calculates Average Precision (AP) at IoU 0.5 and 0.7."
    )
    parser.add_argument(
        "--predictions_file",
        default="results/detections.txt",
        help="Path to the predictions file",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the directory containing images and ground truth (gt.txt)",
    )

    args = parser.parse_args()

    ap05 = compute_ap(args.predictions_file, args.dataset_path, iou=0.5)
    ap07 = compute_ap(args.predictions_file, args.dataset_path, iou=0.7)

    print("\n--- Evaluation Results ---")
    print(f"Dataset:    {args.dataset_path}")
    print(f"File:       {args.predictions_file}")
    print(f"AP@0.5:     {ap05 * 100:.2f}%")
    print(f"AP@0.7:     {ap07 * 100:.2f}%\n")


if __name__ == "__main__":
    main()