"""
Debugging script for the Primary Hough Detector (PanelDetectorHoughPrimary).

This script executes the detection pipeline step-by-step and exports 
intermediate images to a specified output directory to visualize each phase:
  - Preprocessing (CLAHE, Gaussian Blur, Canny)
  - Full-image HoughLinesP detection
  - Horizontal/Vertical line classification and grouping
  - Candidate rectangular bounding box formation
  - HSV blue validation (fast filter)
  - Final result based on mask correlation scoring
"""

import argparse
import os
import cv2
import numpy as np
from src.detector_hough_primary import PanelDetectorHoughPrimary


def debug_detector_hough_primary(img_path: str, output_dir: str = "debug_hough_primary") -> None:
    """
    Executes the Primary Hough detector step-by-step, exporting intermediate phase images.

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
    print(f"\n=== DEBUG HOUGH PRIMARY: {img_name} ({img_w}x{img_h}) ===\n")

    # Instantiate detector
    det = PanelDetectorHoughPrimary()

    # ── PHASE 1: Preprocessing ─────────────────────────────────────────
    print("[1] Preprocessing (CLAHE + GaussianBlur + Canny)...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = det._clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (det._blur_ksize, det._blur_ksize), 1)
    edges = cv2.Canny(blurred, det._canny_low, det._canny_high, apertureSize=3)

    cv2.imwrite(os.path.join(output_dir, "01_gray.png"), gray)
    cv2.imwrite(os.path.join(output_dir, "02_clahe_enhanced.png"), enhanced)
    cv2.imwrite(os.path.join(output_dir, "03_blurred.png"), blurred)
    cv2.imwrite(os.path.join(output_dir, "04_canny_edges.png"), edges)
    print(f"    -> Edge pixels: {np.sum(edges > 0)}")

    # ── PHASE 2: HoughLinesP on full image ─────────────────────────────
    print("[2] Executing HoughLinesP on the full image...")
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=det._hough_threshold,
        minLineLength=det._min_line_length,
        maxLineGap=det._max_line_gap,
    )

    if raw is None:
        print("    -> No lines detected!")
        return

    print(f"    -> Raw lines detected: {len(raw)}")

    # Draw raw lines
    img_lines_raw = img.copy()
    for line in raw:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_lines_raw, (x1, y1), (x2, y2), (0, 255, 255), 1)
    cv2.imwrite(os.path.join(output_dir, "05_hough_lines_raw.png"), img_lines_raw)

    # ── PHASE 3: H/V Classification and Grouping ───────────────────────
    print("[3] Classifying lines into H/V and grouping parallels...")
    h_pos = []
    v_pos = []
    tol = np.deg2rad(det._angle_tol_deg)

    for line in raw:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(abs(y2 - y1), abs(x2 - x1) + 1e-9))

        if angle <= tol:
            h_pos.append((y1 + y2) / 2.0)
        elif angle >= (np.pi / 2.0 - tol):
            v_pos.append((x1 + x2) / 2.0)

    print(f"    -> Raw H positions: {len(h_pos)}")
    print(f"    -> Raw V positions: {len(v_pos)}")

    h_lines = det._group_positions(h_pos)
    v_lines = det._group_positions(v_pos)

    print(f"    -> Grouped H lines: {len(h_lines)} -> {[round(y) for y in h_lines]}")
    print(f"    -> Grouped V lines: {len(v_lines)} -> {[round(x) for x in v_lines]}")

    # Draw grouped lines
    img_lines_grouped = img.copy()
    # H lines (red)
    for y in h_lines:
        cv2.line(img_lines_grouped, (0, int(y)), (img_w, int(y)), (0, 0, 255), 2)
    # V lines (green)
    for x in v_lines:
        cv2.line(img_lines_grouped, (int(x), 0), (int(x), img_h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "06_hough_lines_grouped.png"), img_lines_grouped)

    # ── PHASE 4: Candidate Formulation ─────────────────────────────────
    print("[4] Forming rectangular candidates (H x V intersections)...")
    candidates = det._form_candidates(h_lines, v_lines, img_h, img_w)
    print(f"    -> Candidates after geometric filtering: {len(candidates)}")

    # Draw candidates
    img_candidates = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(candidates):
        cv2.rectangle(img_candidates, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            img_candidates, 
            str(i), 
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 0, 0), 
            1
        )
    cv2.imwrite(os.path.join(output_dir, "07_candidates.png"), img_candidates)

    # ── PHASE 5: HSV Blue Validation ───────────────────────────────────
    print("[5] Validating candidates using HSV blue threshold...")
    blue_filtered = []
    for i, (x1, y1, x2, y2) in enumerate(candidates):
        ratio = det._blue_ratio(img, x1, y1, x2, y2)
        if ratio >= det._min_blue_ratio:
            blue_filtered.append((x1, y1, x2, y2, ratio))
            print(f"    Candidate {i}: blue_ratio={ratio:.2%} [ACCEPTED]")
        else:
            print(f"    Candidate {i}: blue_ratio={ratio:.2%} [REJECTED] (< {det._min_blue_ratio})")

    print(f"\n    -> Passed blue filter: {len(blue_filtered)}/{len(candidates)}")

    # Draw blue filtered candidates
    img_blue_filtered = img.copy()
    for i, (x1, y1, x2, y2, ratio) in enumerate(blue_filtered):
        cv2.rectangle(img_blue_filtered, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_blue_filtered, 
            f"{ratio:.1%}", 
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            1
        )
    cv2.imwrite(os.path.join(output_dir, "08_blue_filtered.png"), img_blue_filtered)

    # ── PHASE 6: Mask Correlation Scoring ──────────────────────────────
    print("[6] Calculating score via ideal mask correlation...")
    final_detections = []
    for i, (x1, y1, x2, y2, ratio) in enumerate(blue_filtered):
        score = det._compute_score(img, x1, y1, x2, y2)
        print(f"    Candidate {i}: ({x1},{y1})-({x2},{y2}) score={score:.3f}")

        if score >= det._score_threshold:
            final_detections.append([x1, y1, x2, y2, score])
            print("        [✓] ACCEPTED")
        else:
            print(f"        [✗] REJECTED (score < {det._score_threshold})")

    # ── PHASE 7: Final Result ──────────────────────────────────────────
    print(f"\n[7] Final Result: {len(final_detections)} detections\n")
    img_final = img.copy()
    for x1, y1, x2, y2, score in final_detections:
        cv2.rectangle(img_final, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_final, 
            f"{score:.2f}", 
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            2
        )
    cv2.imwrite(os.path.join(output_dir, "09_final_detections.png"), img_final)

    print(f"[SUCCESS] Debug images exported to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug script for the Primary Hough Detector.")
    parser.add_argument("--image", type=str, required=True, help="Path to the test image")
    parser.add_argument("--output", type=str, default="debug_hough_primary", help="Output directory for debug images")
    
    args = parser.parse_args()
    debug_detector_hough_primary(args.image, args.output)