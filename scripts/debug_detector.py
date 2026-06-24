"""
Debugging script for the Alternate Hough Detector (PanelDetectorAlt).

This script executes the detection pipeline step-by-step and exports 
intermediate images to a specified output directory to visualize each phase:
  - Raw HSV blue mask
  - Mask after morphological operations (OPEN + CLOSE)
  - Detected contours
  - Per-candidate: Canny edges, HoughLinesP, and final result
"""

import argparse
import os
import cv2
import numpy as np
from src.detector_alt import PanelDetectorAlt


def debug_detector(img_path: str, output_dir: str = "debug") -> None:
    """
    Executes the detector step-by-step, exporting intermediate phase images.

    Args:
        img_path (str): Path to the input test image.
        output_dir (str): Directory where intermediate debug images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Cannot read image at {img_path}")
        return

    img_name = os.path.basename(img_path).replace(".png", "")
    img_h, img_w = img.shape[:2]
    print(f"\n=== DEBUG: {img_name} ({img_w}x{img_h}) ===\n")

    # Instantiate detector
    det = PanelDetectorAlt()

    # ── PHASE 1: Raw HSV Mask ──────────────────────────────────────────
    print("[1] Generating raw HSV blue mask...")
    # Note: Accessing protected member _blue_mask for debugging purposes
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_raw = det._blue_mask(img)
    cv2.imwrite(os.path.join(output_dir, "01_mask_raw.png"), mask_raw)

    # ── PHASE 2: Morphology ────────────────────────────────────────────
    print("[2] Applying morphological operations...")
    mask_morph = det._morphology(mask_raw)
    cv2.imwrite(os.path.join(output_dir, "02_mask_morph.png"), mask_morph)

    # ── PHASE 3: Contours & Geometric Filtering ─────────────────────────
    print("[3] Finding and filtering contours...")
    contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_boxes = det._filter_contours(contours, img_w, img_h)

    img_contours = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(filtered_boxes):
        cv2.rectangle(img_contours, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img_contours, str(i), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imwrite(os.path.join(output_dir, "03_filtered_contours.png"), img_contours)

    # ── PHASE 4 to 6: Per-Candidate Analysis ───────────────────────────
    print(f"[4-6] Analyzing {len(filtered_boxes)} candidates using Hough...")
    final_detections = []

    for i, (x1, y1, x2, y2) in enumerate(filtered_boxes):
        print(f"  -> Candidate {i}: Box({x1}, {y1}, {x2}, {y2})")
        subdir = os.path.join(output_dir, f"cand_{i:02d}")
        os.makedirs(subdir, exist_ok=True)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        cv2.imwrite(os.path.join(subdir, "a_roi_rgb.png"), roi)

        # Canny Edges
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        cv2.imwrite(os.path.join(subdir, "b_canny.png"), edges)

        # Hough Lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=10)
        
        roi_lines = roi.copy()
        h_count = 0
        v_count = 0

        if lines is not None:
            tol = 15.0
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                angle = np.abs(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))
                
                if angle <= tol:
                    cv2.line(roi_lines, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
                    h_count += 1
                elif angle >= (90.0 - tol):
                    cv2.line(roi_lines, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
                    v_count += 1

        cv2.imwrite(os.path.join(subdir, "c_hough_lines.png"), roi_lines)
        print(f"      H-Lines: {h_count}, V-Lines: {v_count}")

        # Score computation
        score = det._compute_score(img, x1, y1, x2, y2)
        print(f"      Score: {score:.3f}")

        if score >= det._score_threshold:
            final_detections.append([x1, y1, x2, y2, score])
            print("      [✓] ACCEPTED")
        else:
            print(f"      [✗] REJECTED (score < {det._score_threshold})")
        print("")

    # ── PHASE 7: Final Result ──────────────────────────────────────────
    print(f"[7] Final Result: {len(final_detections)} detections\n")
    img_final = img.copy()
    for x1, y1, x2, y2, score in final_detections:
        cv2.rectangle(img_final, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_final, f"{score:.3f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imwrite(os.path.join(output_dir, "07_final_result.png"), img_final)
    print(f"Debug completed. Visualizations saved to '{output_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug script for the Alternate Hough Detector.")
    parser.add_argument("--image", type=str, required=True, help="Path to the test image")
    parser.add_argument("--output", type=str, default="debug", help="Output directory for debug images")
    
    args = parser.parse_args()
    debug_detector(args.image, args.output)