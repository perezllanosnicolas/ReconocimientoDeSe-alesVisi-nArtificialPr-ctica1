"""
MSER-based Highway Panel Detector.

Pipeline:
  1. Maximally Stable Extremal Regions (MSER) to detect high-contrast areas (text/borders).
  2. Geometric filtering of bounding boxes (size, aspect ratio).
  3. Bounding box expansion (padding) to capture the full blue background.
  4. HSV blue validation and morphological correlation scoring against an ideal mask.
"""

import cv2
import numpy as np
from typing import List, Tuple


class PanelDetectorMSER:
    """
    Detects information panels using MSER (Maximally Stable Extremal Regions)
    to identify areas with high contrast, followed by geometric and color validation.
    """

    def __init__(self, min_area: int = 500, hue_min: int = 90, hue_max: int = 130, 
                 sat_min: int = 150, score_threshold: float = 0.2, delta: int = 2,
                 mask_height: int = 40, mask_width: int = 80, min_box_area: int = 1000, 
                 min_box_width: int = 18, min_box_height: int = 18, 
                 min_aspect_ratio: float = 0.6, max_aspect_ratio: float = 4.0) -> None:
        
        # --- MSER Parameters ---
        self.min_area = min_area
        self.delta = delta
        self.mser = cv2.MSER_create(delta=self.delta, min_area=self.min_area, max_area=205000)
        
        # --- Geometric Filtering Parameters ---
        self.min_box_area = min_box_area
        self.min_box_width = min_box_width
        self.min_box_height = min_box_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        
        # --- Color Validation & Scoring Parameters ---
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.score_threshold = score_threshold
        
        self.mask_height = mask_height
        self.mask_width = mask_width
        
        # Blue HSV range (Strictly matching the original [hue, sat, 50])
        self.blue_lower = np.array([self.hue_min, self.sat_min, 50], dtype=np.uint8)
        self.blue_upper = np.array([self.hue_max, 255, 255], dtype=np.uint8)
        
        # Ideal validation mask (Calculated once at initialization)
        self.ideal_mask = self._create_ideal_mask(self.mask_height, self.mask_width)

    def _create_ideal_mask(self, h: int, w: int) -> np.ndarray:
        """Creates an ideal synthetic mask with a -2 penalty border exactly as original."""
        mask = np.ones((h, w), dtype=np.float32)
        # Original code set a 4-pixel border to heavily penalize over-segmentation
        mask[0:4, :] = -2.0
        mask[-4:, :] = -2.0
        mask[:, 0:4] = -2.0
        mask[:, -4:] = -2.0
        return mask

    def _expand_bbox(self, x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Expands the bounding box by 4.5% to capture the outer border."""
        exp_w = int(w * 0.045)
        exp_h = int(h * 0.045)
        
        nx1 = max(0, x - exp_w)
        ny1 = max(0, y - exp_h)
        nx2 = min(img_w, x + w + exp_w)
        ny2 = min(img_h, y + h + exp_h)
        
        return nx1, ny1, nx2, ny2

    def _compute_score(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculates candidate confidence using ideal mask correlation."""
        roi = image[y1:y2, x1:x2]
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            return 0.0
            
        roi_resized = cv2.resize(roi, (self.mask_width, self.mask_height))
        roi_hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(roi_hsv, self.blue_lower, self.blue_upper)
        
        # Normalize to [0, 1]
        mask_blue_norm = mask_blue.astype(np.float32) / 255.0
        
        # Mathematical correlation against the strict ideal mask (with -2 penalty)
        correlation_matrix = mask_blue_norm * self.ideal_mask
        correlation = float(np.sum(correlation_matrix))
        
        max_possible = float(np.sum(self.ideal_mask[self.ideal_mask > 0]))
        score = correlation / max_possible if max_possible > 0 else 0.0
        
        # Bound the score to [0, 1] interval
        return float(np.clip(score, 0.0, 1.0))

    def detect(self, image: np.ndarray) -> List[List[float]]:
        """Main pipeline execution for the MSER detector."""
        img_h, img_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Detect MSER regions
        regions, bboxes = self.mser.detectRegions(gray)
        if len(bboxes) == 0:
            return []
            
        detections = []
        
        # 2. Process each detected bounding box
        for box in bboxes:
            x, y, w, h = box
            
            # Geometric Filters
            if w < self.min_box_width or h < self.min_box_height:
                continue
                
            area = w * h
            if area < self.min_box_area:
                continue
                
            aspect_ratio = float(w) / h
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
                
            # 3. Expand bounding box
            nx1, ny1, nx2, ny2 = self._expand_bbox(x, y, w, h, img_w, img_h)
            
            # 4. Color Validation & Score
            score = self._compute_score(image, nx1, ny1, nx2, ny2)
            
            if score >= self.score_threshold:
                detections.append([float(nx1), float(ny1), float(nx2), float(ny2), score])
                
        return detections