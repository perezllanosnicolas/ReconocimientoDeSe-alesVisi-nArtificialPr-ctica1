"""
Alternate Hough-based Highway Sign Detector.

Pipeline:
  1. Saturated blue HSV mask + morphology -> candidate blobs.
  2. Geometric filtering (size, aspect ratio, solidity).
  3. Bounded bounding box expansion to include the white outer border.
  4. HoughLinesP within the ROI: confirms rectangular structure (H + V edges)
     and modulates the score without altering the coordinates.
  5. Score = Correlation (F1 + specificity) with an ideal 40x80 blue mask,
     multiplied by the Hough confidence factor [0.8 - 1.0].
"""

import cv2
import numpy as np
from typing import List, Tuple


class PanelDetectorAlt:
    """
    Detects blue rectangular highway panels by combining color segmentation
    with the Probabilistic Hough Transform (HoughLinesP).

    Saturated blue locates the candidates; HoughLinesP confirms that
    the ROI has the structural rectangular edges of a traffic panel.
    """

    def __init__(self) -> None:
        # --- Blue Color Mask Parameters (HSV Space) ---
        self._blue_lower = np.array([100, 200,  70], dtype=np.uint8)
        self._blue_upper = np.array([128, 255, 255], dtype=np.uint8)

        # --- Morphological Parameters ---
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

        # --- Geometric Constraints ---
        self._min_area = 1500
        self._max_area = 150000
        self._min_aspect = 0.4
        self._max_aspect = 4.0
        self._min_solidity = 0.50

        # --- HoughLinesP Parameters ---
        self._hough_thresh = 30
        self._min_line_len = 20
        self._max_line_gap = 10
        self._angle_tol = 15.0  # Degrees of tolerance for Horizontal/Vertical lines

        # --- Scoring Parameters ---
        self._score_threshold = 0.20
        self._ideal_mask = self._create_ideal_mask(40, 80)

    def _create_ideal_mask(self, h: int, w: int) -> np.ndarray:
        """
        Creates an ideal synthetic mask (white border, blue interior)
        normalized to [0, 1] for correlation scoring.
        """
        mask = np.zeros((h, w), dtype=np.float32)
        # Assuming a ~5% outer margin for the border
        m_x = int(w * 0.05)
        m_y = int(h * 0.05)
        mask[m_y:(h - m_y), m_x:(w - m_x)] = 1.0
        return mask

    def _blue_mask(self, img: np.ndarray) -> np.ndarray:
        """Extracts the raw blue HSV mask."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self._blue_lower, self._blue_upper)

    def _morphology(self, mask: np.ndarray) -> np.ndarray:
        """Applies opening (noise removal) and closing (gap filling)."""
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._kernel_close)
        return closed

    def _filter_contours(self, contours: List[np.ndarray], img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
        """Filters contours based on area, solidity, and aspect ratio."""
        valid_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area or area > self._max_area:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
                
            solidity = area / hull_area
            if solidity < self._min_solidity:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / h
            if aspect < self._min_aspect or aspect > self._max_aspect:
                continue

            valid_boxes.append((x, y, x + w, y + h))
            
        return valid_boxes

    def _expand_bbox(self, x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Expands the bounding box slightly to capture the white border."""
        w = x2 - x1
        h = y2 - y1
        pad_x = int(w * 0.08)
        pad_y = int(h * 0.12)

        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(img_w - 1, x2 + pad_x)
        ny2 = min(img_h - 1, y2 + pad_y)

        return nx1, ny1, nx2, ny2

    def _compute_score(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        Computes the final confidence score combining structural Hough validation
        and morphological mask correlation.
        """
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        roi_h, roi_w = roi.shape[:2]
        
        # 1. Structural Verification (HoughLinesP)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 
            threshold=self._hough_thresh, 
            minLineLength=self._min_line_len, 
            maxLineGap=self._max_line_gap
        )

        h_count = 0
        v_count = 0
        if lines is not None:
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                angle = np.abs(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))
                if angle <= self._angle_tol:
                    h_count += 1
                elif angle >= (90.0 - self._angle_tol):
                    v_count += 1

        # Penalize lack of structural edges
        hough_factor = 1.0
        if h_count == 0 and v_count == 0:
            hough_factor = 0.80
        elif h_count == 0 or v_count == 0:
            hough_factor = 0.90

        # 2. Appearance Verification (Mask Correlation)
        roi_resized = cv2.resize(roi, (80, 40))
        roi_hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
        
        # Broaden blue range slightly for resized/interpolated ROI
        lower_b = np.array([90, 100, 40], dtype=np.uint8)
        upper_b = np.array([135, 255, 255], dtype=np.uint8)
        mask_roi = cv2.inRange(roi_hsv, lower_b, upper_b)

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
        f1 = (2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
        specificity = tn / negatives if negatives > 0 else 0.0

        # Final weighted score
        score_base = 0.55 * f1 + 0.45 * specificity
        return float(np.clip(score_base * hough_factor, 0.0, 1.0))

    def detect(self, image: np.ndarray) -> List[List[float]]:
        """
        Main detection pipeline for a given BGR image.

        Returns:
            List[List[float]]: A list of detections in the format [x1, y1, x2, y2, score].
        """
        img_h, img_w = image.shape[:2]

        mask = self._blue_mask(image)
        mask = self._morphology(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_boxes = self._filter_contours(contours, img_w, img_h)

        detections = []
        for (x1, y1, x2, y2) in valid_boxes:
            nx1, ny1, nx2, ny2 = self._expand_bbox(x1, y1, x2, y2, img_w, img_h)
            score = self._compute_score(image, nx1, ny1, nx2, ny2)

            if score >= self._score_threshold:
                detections.append([float(nx1), float(ny1), float(nx2), float(ny2), score])

        return detections