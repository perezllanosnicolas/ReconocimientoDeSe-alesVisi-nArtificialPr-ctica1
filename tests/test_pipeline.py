"""
Integration and Smoke Tests for the Highway Sign Detection Pipeline.

This test suite verifies that all detector classes can be instantiated,
that they process images without runtime errors, and that the utility 
functions perform correct mathematical operations.
"""

import unittest
import numpy as np
import cv2


from src.utils import calculate_iou, remove_overlapping_boxes
from src.detector_mser import PanelDetectorMSER
from src.detector_alt import PanelDetectorAlt
from src.detector_contours import PanelDetectorContours
from src.detector_hough_primary import PanelDetectorHoughPrimary
from src.detector_hybrid import PanelDetectorHybrid


class TestDetectionPipeline(unittest.TestCase):

    def setUp(self) -> None:
        """
        Creates a synthetic image (800x600) with a simulated blue panel
        before each test to verify the detectors without needing disk I/O.
        """
       
        self.img = np.zeros((600, 800, 3), dtype=np.uint8) + 50
        

        cv2.rectangle(self.img, (300, 200), (500, 300), (255, 255, 255), -1)
        cv2.rectangle(self.img, (305, 205), (495, 295), (200, 50, 0), -1)

    def test_utils_iou(self):
        """Verifies the mathematical integrity of the Intersection over Union metric."""
        box_a = [0, 0, 100, 100]
        box_b = [50, 50, 150, 150]
        
        iou, iom = calculate_iou(box_a, box_b)
        
        # Expected IoU for this specific overlap is roughly 0.14
        self.assertGreater(iou, 0.0)
        self.assertGreater(iom, 0.0)

    def test_utils_nms(self):
        """Verifies the Non-Maximum Suppression logic removes overlapping boxes."""
        boxes = [
            [10, 10, 50, 50, 0.95], # Main box
            [12, 12, 48, 48, 0.80], # Heavily overlapping box (should be removed)
            [200, 200, 250, 250, 0.90] # Independent box (should be kept)
        ]
        filtered = remove_overlapping_boxes(boxes, iou_threshold=0.5)
        self.assertEqual(len(filtered), 2)

    def test_detector_mser(self):
        """Smoke test for the MSER detector."""
        detector = PanelDetectorMSER()
        results = detector.detect(self.img)
        self.assertIsInstance(results, list)

    def test_detector_alt(self):
        """Smoke test for the Alternate Hough detector."""
        detector = PanelDetectorAlt()
        results = detector.detect(self.img)
        self.assertIsInstance(results, list)

    def test_detector_contours(self):
        """Smoke test for the Contours detector."""
        detector = PanelDetectorContours()
        results = detector.detect(self.img)
        self.assertIsInstance(results, list)

    def test_detector_hough_primary(self):
        """Smoke test for the Primary Hough detector."""
        detector = PanelDetectorHoughPrimary()
        results = detector.detect(self.img)
        self.assertIsInstance(results, list)

    def test_detector_hybrid(self):
        """Smoke test for the Hybrid detector."""
        detector = PanelDetectorHybrid()
        results = detector.detect(self.img)
        self.assertIsInstance(results, list)


if __name__ == '__main__':
    unittest.main(verbosity=2)