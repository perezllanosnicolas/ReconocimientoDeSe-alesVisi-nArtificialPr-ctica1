"""
Contours-based Highway Panel Detector.

An alternate detection module based on Chromatic Segmentation (HSV), 
Connected Component Analysis, and Strict Geometric Filtering.
Features robust scoring via Asymmetric Padding and Text Eraser morphology.
"""

import cv2
import numpy as np
from typing import List


class PanelDetectorContours: 
    """
    Detects highway information panels using a purely classical Computer Vision 
    pipeline relying on HSV segmentation and topology checks.
    """
    
    def __init__(self, hue_min: int = 95, hue_max: int = 125, sat_min: int = 150, 
                 val_min: int = 45, score_threshold: float = 0.2, mask_height: int = 40, 
                 mask_width: int = 80, min_box_area: int = 2500, max_box_area: int = 150000, 
                 min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 4.5,
                 min_solidity: float = 0.40, pad_w: float = 0.05, pad_h: float = 0.08, 
                 edge_penalty: float = -2.0) -> None:
        
        # --- Color Parameters (HSV Space) ---
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.val_min = val_min
        
        # --- Geometric Parameters (Shape & Scale) ---
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_solidity = min_solidity
        
        # --- Topological & Scoring Parameters ---
        self.score_threshold = score_threshold
        self.mask_height = mask_height
        self.mask_width = mask_width
        self.pad_w = pad_w
        self.pad_h = pad_h
        self.edge_penalty = edge_penalty
        
        # --- Ideal Correlation Mask Initialization ---
        # The ideal mask represents a perfect panel: a blue rectangle with a white border.
        self.ideal_mask = np.zeros((self.mask_height, self.mask_width), dtype=np.float32)
        
        # Asymmetric padding to center the blue mass
        border_x = int(self.mask_width * self.pad_w)
        border_y = int(self.mask_height * self.pad_h)
        self.ideal_mask[border_y:(self.mask_height - border_y), border_x:(self.mask_width - border_x)] = 1.0

        # Outer border penalty to heavily penalize over-segmentation
        self.ideal_mask[0:border_y, :] = self.edge_penalty
        self.ideal_mask[(self.mask_height - border_y):, :] = self.edge_penalty
        self.ideal_mask[:, 0:border_x] = self.edge_penalty
        self.ideal_mask[:, (self.mask_width - border_x):] = self.edge_penalty

    def detect(self, image: np.ndarray) -> List[List[float]]:
        """
        Executes the detection pipeline on a BGR image.
        
        Args:
            image (np.ndarray): Input image in BGR format.
            
        Returns:
            List[List[float]]: Detected panels as a list of bounding boxes 
                               with their associated score [x1, y1, x2, y2, score].
        """
        detections = []
        img_h, img_w = image.shape[:2]

        # 1. Color Segmentation (HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([self.hue_min, self.sat_min, self.val_min])
        upper_blue = np.array([self.hue_max, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 2. Morphological Pre-processing
        # Opening removes small noise, Closing connects fragmented panel parts
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((7, 7), np.uint8)
        
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_open)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_close)

        # 3. Connected Components Extraction
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Area Filter
            if area < self.min_box_area or area > self.max_box_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Aspect Ratio Filter (Discards vertical pillars / long horizontal barriers)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Solidity Filter (Discards irregular shapes like cars or non-convex objects)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                if solidity < self.min_solidity:
                    continue

            # Candidate accepted for deep evaluation
            candidate_boxes.append((x, y, w, h))

        # 4. Score Calculation via Correlation
        for (x, y, w, h) in candidate_boxes:
            # Expand bounding box slightly to capture the white border
            pad_x = int(w * self.pad_w)
            pad_y = int(h * self.pad_h)
            
            x_min = max(0, x - pad_x)
            y_min = max(0, y - pad_y)
            x_max = min(img_w, x + w + pad_x)
            y_max = min(img_h, y + h + pad_y)
            
            roi = image[y_min:y_max, x_min:x_max]
            
            if roi.size == 0:
                continue
                
            # Resize ROI to match ideal mask dimensions
            roi_resized = cv2.resize(roi, (self.mask_width, self.mask_height))
            roi_hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
            mask_roi_blue = cv2.inRange(roi_hsv, lower_blue, upper_blue)
            
            # Internal Morphological Closing (Text Eraser): 
            # Maximizes blue density by engulfing gaps created by white text.
            kernel_text = np.ones((5, 5), np.uint8)
            mask_roi_blue = cv2.morphologyEx(mask_roi_blue, cv2.MORPH_CLOSE, kernel_text)

            # Binarized Normalization [0, 1]
            mask_blue_norm = mask_roi_blue.astype(np.float32) / 255.0
            
            # Mathematical correlation against the strict ideal mask
            correlation_matrix = mask_blue_norm * self.ideal_mask
            correlation = np.sum(correlation_matrix)
            
            # Calculate maximum possible score to normalize output
            max_possible = np.sum(self.ideal_mask[self.ideal_mask > 0])
            score = float(correlation / max_possible) if max_possible > 0 else 0.0
            
            # Bound the score to [0, 1] interval
            score = max(0.0, min(1.0, score))
            
            if score >= self.score_threshold:
                detections.append([float(x_min), float(y_min), float(x_max), float(y_max), score])

        return detections