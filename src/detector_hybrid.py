"""
Hybrid Highway Panel Detector.

Combines the strengths of two paradigms:
  1. HSV Blue Mask (Primary Detector): Highly specific, finds panel locations.
  2. HoughLinesP (Refiner): Highly precise, finds the exact structural borders (H/V lines).

Pipeline:
  1. HSV blue extraction -> morphological cleanup -> candidate blobs.
  2. Geometric filtering.
  3. Expand search zone around each candidate.
  4. HoughLinesP within the expanded zone -> snap boundaries to white edges.
  5. Refine bbox using these lines.
  6. Score by morphological mask correlation.
"""

import cv2
import numpy as np
from typing import List, Tuple


class PanelDetectorHybrid:
    """
    Hybrid detector using color segmentation to find regions of interest,
    and the Probabilistic Hough Transform to snap bounding boxes to real edges.
    """

    def __init__(self) -> None:
        # --- Primary Detector: HSV Blue Mask ---
        self._blue_lower = np.array([100, 200, 70], dtype=np.uint8)
        self._blue_upper = np.array([128, 255, 255], dtype=np.uint8)

        # --- Morphology for Blue Mask Cleanup ---
        self._morph_open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._morph_close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        # --- Geometric Constraints ---
        self._min_blob_area = 200
        self._min_blob_width = 15
        self._min_blob_height = 10
        self._min_aspect = 0.5
        self._max_aspect = 8.0

        # --- Search Zone Expansion ---
        self._search_expand_px = 35

        # --- Refiner: HoughLinesP ---
        self._canny_low = 50
        self._canny_high = 150
        self._hough_threshold = 15
        self._min_line_ratio = 0.18
        self._max_gap_ratio = 0.08
        self._angle_tol_deg = 20.0

        # --- Scoring Parameters ---
        self._mask_h = 40
        self._mask_w = 80
        self._score_threshold = 0.25
        self._ideal_mask = self._create_ideal_mask()

    def _create_ideal_mask(self) -> np.ndarray:
        """Creates the ideal mask based on the exact original mathematical ratio."""
        mask = np.zeros((self._mask_h, self._mask_w), dtype=np.uint8)
        by = int(self._mask_h * 0.20)  # 20% vertical padding
        bx = int(self._mask_w * 0.16)  # 16% horizontal padding
        mask[by : self._mask_h - by, bx : self._mask_w - bx] = 1
        return mask

    def _blue_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Extracts and morphologically cleans the blue HSV mask."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._blue_lower, self._blue_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_close_k, iterations=2)
        return mask

    def _extract_blob_candidates(self, mask: np.ndarray, img_h: int, img_w: int) -> List[Tuple]:
        """Finds contours and generates expanded search zones for Hough refinement."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        ep = self._search_expand_px

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < self._min_blob_width or h < self._min_blob_height:
                continue
            if w * h < self._min_blob_area:
                continue
            
            aspect = w / float(h)
            if not (self._min_aspect <= aspect <= self._max_aspect):
                continue

            # Inner blob
            bx1, by1 = x, y
            bx2, by2 = x + w, y + h

            # Expanded search zone
            sx1 = max(0, bx1 - ep)
            sy1 = max(0, by1 - ep)
            sx2 = min(img_w, bx2 + ep)
            sy2 = min(img_h, by2 + ep)

            candidates.append((sx1, sy1, sx2, sy2, (bx1, by1, bx2, by2)))

        return candidates

    def _refine_with_hough(self, img_bgr: np.ndarray, sx1: int, sy1: int, sx2: int, sy2: int, blob: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Snaps the bounding box to the nearest external structural edges."""
        roi = img_bgr[sy1:sy2, sx1:sx2]
        roi_h, roi_w = roi.shape[:2]

        if roi.size == 0:
            return blob

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self._canny_low, self._canny_high)

        min_len = max(8, int(min(roi_w, roi_h) * self._min_line_ratio))
        max_gap = max(3, int(min(roi_w, roi_h) * self._max_gap_ratio))

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=self._hough_threshold, minLineLength=min_len, maxLineGap=max_gap
        )

        bx1_roi = blob[0] - sx1
        by1_roi = blob[1] - sy1
        bx2_roi = blob[2] - sx1
        by2_roi = blob[3] - sy1

        # Original Fallback Safety Net
        fallback_px = 10
        fy1_roi = max(0, by1_roi - fallback_px)
        fy2_roi = min(roi_h, by2_roi + fallback_px)
        fx1_roi = max(0, bx1_roi - fallback_px)
        fx2_roi = min(roi_w, bx2_roi + fallback_px)

        if lines is not None:
            tol = np.deg2rad(self._angle_tol_deg)
            h_lines = []
            v_lines = []

            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                angle = abs(np.arctan2(abs(ly2 - ly1), abs(lx2 - lx1) + 1e-9))

                if angle <= tol:
                    y_mid = (ly1 + ly2) / 2.0
                    h_lines.append(y_mid)
                elif angle >= (np.pi / 2.0 - tol):
                    x_mid = (lx1 + lx2) / 2.0
                    v_lines.append(x_mid)

            if h_lines:
                h_above = [y for y in h_lines if y < by1_roi - 5]
                h_below = [y for y in h_lines if y > by2_roi + 5]

                if h_above:
                    fy1_roi = max(0, int(max(h_above)))
                if h_below:
                    fy2_roi = min(roi_h, int(min(h_below)))

            if v_lines:
                v_left = [x for x in v_lines if x < bx1_roi - 5]
                v_right = [x for x in v_lines if x > bx2_roi + 5]

                if v_left:
                    fx1_roi = max(0, int(max(v_left)))
                if v_right:
                    fx2_roi = min(roi_w, int(min(v_right)))

        fx1 = sx1 + fx1_roi
        fy1 = sy1 + fy1_roi
        fx2 = sx1 + fx2_roi
        fy2 = sy1 + fy2_roi

        return fx1, fy1, fx2, fy2

    def _compute_score(self, img_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculates candidate confidence using the strict original mask correlation."""
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        resized = cv2.resize(roi, (self._mask_w, self._mask_h), interpolation=cv2.INTER_AREA)
        hsv_roi = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # Strict original color range
        M = (cv2.inRange(hsv_roi, self._blue_lower, self._blue_upper) > 0).astype(np.uint8)

        ideal = self._ideal_mask
        tp = float(np.sum((M == 1) & (ideal == 1)))
        fp = float(np.sum((M == 1) & (ideal == 0)))
        tn = float(np.sum((M == 0) & (ideal == 0)))
        positives = float(np.sum(ideal == 1))
        negatives = float(np.sum(ideal == 0))

        if positives <= 0:
            return 0.0

        recall = tp / positives
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / negatives if negatives > 0 else 0.0

        return float(np.clip(0.55 * f1 + 0.45 * specificity, 0.0, 1.0))

    def detect(self, image: np.ndarray) -> List[List[float]]:
        """Main pipeline execution for the Hybrid detector."""
        img_h, img_w = image.shape[:2]

        mask = self._blue_mask(image)
        candidates = self._extract_blob_candidates(mask, img_h, img_w)

        detections = []
        for (sx1, sy1, sx2, sy2, blob) in candidates:
            fx1, fy1, fx2, fy2 = self._refine_with_hough(image, sx1, sy1, sx2, sy2, blob)
            score = self._compute_score(image, fx1, fy1, fx2, fy2)

            if score >= self._score_threshold:
                detections.append([float(fx1), float(fy1), float(fx2), float(fy2), score])

        return detections