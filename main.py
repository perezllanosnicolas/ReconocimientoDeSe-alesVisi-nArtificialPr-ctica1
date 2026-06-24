"""
Main entry point for the Highway Traffic Sign Detection pipeline.

This script loads images from a specified directory, processes them using
the selected computer vision detector, applies Non-Maximum Suppression (NMS)
to filter overlapping bounding boxes, and exports the results both as 
annotated images and a standardized text file.
"""

import argparse
import os
import glob
import cv2

from src.detector_mser import PanelDetectorMSER
from src.detector_alt import PanelDetectorAlt
from src.detector_hough_primary import PanelDetectorHoughPrimary
from src.detector_hybrid import PanelDetectorHybrid
from src.utils import remove_overlapping_boxes


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing paths and configuration.
    """
    parser = argparse.ArgumentParser(description="Highway Traffic Sign Detector")

    # Note: train_path is kept for compatibility, although it is not directly used in inference.
    parser.add_argument("--train_path", type=str, required=False, help="Path to the training directory")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the testing directory")
    parser.add_argument(
        "--detector", 
        type=str, 
        required=True, 
        choices=['mser', 'hough', 'hough_primary', 'hybrid'], 
        help="Name of the detection algorithm to use"
    )

    # Best practice: Avoid hardcoded paths by passing them as arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/images", 
        help="Directory to save annotated images"
    )
    parser.add_argument(
        "--output_txt", 
        type=str, 
        default="results/detections.txt", 
        help="Path to save the text results file"
    )

    return parser.parse_args()


def main() -> None:
    """
    Main execution function for the detection pipeline.
    """
    args = parse_args()

    # 1. Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Make sure the parent directory of the text file exists
    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)

    # 2. Instantiate the selected detector
    if args.detector == 'mser':
        print("Using MSER detector...")
        detector = PanelDetectorMSER()
    elif args.detector == 'hough':
        print("Using Hough detector (color first)...")
        detector = PanelDetectorAlt()
    elif args.detector == 'hough_primary':
        print("Using Hough detector (Hough first)...")
        detector = PanelDetectorHoughPrimary()
    elif args.detector == 'hybrid':
        print("Using Hybrid detector (color + Hough refinement)...")
        detector = PanelDetectorHybrid()

    # 3. Process test images
    test_images = glob.glob(os.path.join(args.test_path, "*.png"))

    if not test_images:
        print(f"Warning: No PNG images found in '{args.test_path}'.")
        return

    with open(args.output_txt, 'w') as f_out: 
        for img_path in test_images:
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Error loading image: {img_name}")
                continue
    
            # Run the selected detection algorithm
            bbox_list = detector.detect(img)

            # Filter overlapping bounding boxes (NMS)
            # A very low threshold is used to remove almost any minimal overlap.
            bbox_list = remove_overlapping_boxes(
                bbox_list, 
                iou_threshold=0.2, 
                iom_threshold=0.999
            ) 
    
            # Draw and save the results
            img_result = img.copy()
            for bbox in bbox_list:
                # Unpack the bounding box and score
                x1, y1, x2, y2, score = bbox
        
                x1_int, y1_int = int(x1), int(y1)
                x2_int, y2_int = int(x2), int(y2)
            
                # OUTPUT FORMAT: <filename>;<x1>;<y1>;<x2>;<y2>;<class_type>;<score>
                # Class type is always 1 for this project
                f_out.write(f"{img_name};{x1_int};{y1_int};{x2_int};{y2_int};1;{score:.3f}\n")
    
                # Draw red bounding box and yellow text score
                cv2.rectangle(img_result, (x1_int, y1_int), (x2_int, y2_int), (0, 0, 255), 2)
                cv2.putText(
                    img_result, 
                    f"{score:.3f}", 
                    (x1_int, y1_int - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 255), 
                    2
                )

            # Save the annotated image
            cv2.imwrite(os.path.join(args.output_dir, img_name), img_result)

    print(f"Processing complete. Results saved in '{args.output_dir}/' and '{args.output_txt}'.")


if __name__ == "__main__":
    main()