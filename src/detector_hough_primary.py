"""
Primary Hough-based Highway Sign Detector.

Pipeline:
  1. Preprocessing (CLAHE + GaussianBlur + Canny) on the full image.
  2. HoughLinesP to detect structural lines across the image.
  3. Group parallel lines to form horizontal and vertical axes.
  4. Form candidate bounding boxes at the intersections of H x V lines.
  5. Validate candidates using an HSV blue saturation threshold.
  6. Score candidates via correlation with an ideal 40x80 mask.
"""

import cv2
import numpy as np
from typing import List, Tuple


class PanelDetectorHoughPrimary:
    """
    Detects rectangular highway panels using HoughLinesP as the primary 
    proposal mechanism, followed by color and morphological validation.
    """

    def __init__(self) -> None:
        # --- Preprocessing Parameters ---
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._blur_ksize = 5
        self._canny_low = 40
        self._canny_high = 120

        # --- HoughLinesP Parameters ---
        self._hough_threshold = 40
        self._min_line_length = 30
        self._max_line_gap = 10
        self._angle_tol_deg = 15.0
        self._group_dist_tol = 20.0

        # --- Geometric Constraints ---
        self._min_w = 20
        self._min_h = 20
        self._min_aspect = 0.5
        self._max_aspect = 4.0
        self._min_area = 1500
        self._max_area = 150000

        # --- Color Validation Parameters (HSV Space) ---
        self._blue_lower = np.array([90, 100, 40], dtype=np.uint8)
        self._blue_upper = np.array([135, 255, 255], dtype=np.uint8)
        self._min_blue_ratio = 0.15

        # --- Scoring Parameters ---
        self._score_threshold = 0.20
        self._ideal_mask = self._create_ideal_mask(40, 80)

    def _create_ideal_mask(self, h: int, w: int) -> np.ndarray:
        """Creates an ideal synthetic mask (white border, blue interior)."""
        mask = np.zeros((h, w), dtype=np.float32)
        m_x = int(w * 0.05)
        m_y = int(h * 0.05)
        mask[m_y:(h - m_y), m_x:(w - m_x)] = 1.0
        return mask

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Applies CLAHE, blur, and Canny edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (self._blur_ksize, self._blur_ksize), 1)
        edges = cv2.Canny(blurred, self._canny_low, self._canny_high, apertureSize=3)
        return edges

    def _detect_hough_lines(self, image: np.ndarray) -> Tuple[List[float], List[float]]:
        """Extracts and classifies Horizontal and Vertical lines."""
        edges = self._preprocess_image(image)
        
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi / 180, 
            threshold=self._hough_threshold,
            minLineLength=self._min_line_length, 
            maxLineGap=self._max_line_gap
        )

        h_positions = []
        v_positions = []

        if lines is not None:
            tol_rad = np.deg2rad(self._angle_tol_deg)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(abs(y2 - y1), abs(x2 - x1) + 1e-9))

                if angle <= tol_rad:
                    h_positions.append(float(y1 + y2) / 2.0)
                elif angle >= (np.pi / 2.0 - tol_rad):
                    v_positions.append(float(x1 + x2) / 2.0)

        h_lines_grouped = self._group_positions(h_positions)
        v_lines_grouped = self._group_positions(v_positions)

        return h_lines_grouped, v_lines_grouped

    def _group_positions(self, positions: List[float]) -> List[float]:
        """Groups nearby parallel lines into a single representative axis."""
        if not positions:
            return []

        positions.sort()
        grouped = []
        current_group = [positions[0]]

        for pos in positions[1:]:
            if pos - current_group[-1] <= self._group_dist_tol:
                current_group.append(pos)
            else:
                grouped.append(float(np.mean(current_group)))
                current_group = [pos]
        
        grouped.append(float(np.mean(current_group)))
        return grouped

    def _form_candidates(self, h_lines: List[float], v_lines: List[float], img_h: int, img_w: int) -> List[Tuple[int, int, int, int]]:
        """Forms rectangular bounding boxes from intersecting H and V axes."""
        candidates = []
        for i in range(len(v_lines)):
            for j in range(i + 1, len(v_lines)):
                x1, x2 = int(v_lines[i]), int(v_lines[j])
                w = x2 - x1
                if w < self._min_w:
                    continue

                for k in range(len(h_lines)):
                    for m in range(k + 1, len(h_lines)):
                        y1, y2 = int(h_lines[k]), int(h_lines[m])
                        h = y2 - y1
                        if h < self._min_h:
                            continue

                        area = w * h
                        if area < self._min_area or area > self._max_area:
                            continue

                        aspect = float(w) / h
                        if aspect < self._min_aspect or aspect > self._max_aspect:
                            continue

                        candidates.append((x1, y1, x2, y2))
                        
        return candidates

    def _blue_ratio(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculates the ratio of blue pixels in the candidate ROI."""
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._blue_lower, self._blue_upper)
        return float(np.sum(mask > 0) / (roi.shape[0] * roi.shape[1]))

    def _compute_score(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculates candidate confidence using ideal mask correlation."""
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        roi_resized = cv2.resize(roi, (80, 40))
        roi_hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
        mask_roi = cv2.inRange(roi_hsv, self._blue_lower, self._blue_upper)

        # Normalize to [0, 1]
        M = (mask_roi > 0).astype(np.float32)
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
        """Main pipeline execution for the Primary Hough detector."""
        img_h, img_w = image.shape[:2]

        h_lines, v_lines = self._detect_hough_lines(image)
        if len(h_lines) < 2 or len(v_lines) < 2:
            return []

        candidates = self._form_candidates(h_lines, v_lines, img_h, img_w)

        detections = []
        for (x1, y1, x2, y2) in candidates:
            if self._blue_ratio(image, x1, y1, x2, y2) < self._min_blue_ratio:
                continue

            score = self._compute_score(image, x1, y1, x2, y2)
            if score >= self._score_threshold:
                detections.append([float(x1), float(y1), float(x2), float(y2), score])

        return detections