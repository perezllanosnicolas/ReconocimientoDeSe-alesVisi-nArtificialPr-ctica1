# Comparative Analysis of Highway Traffic Sign Detectors

**Authors:** Nicolás Pérez Llanos and Rubén Pisonero Cuenca  
**Context:** Computer Vision Project (Computer Engineering)

---
        
## Performance Summary

The following table summarizes the performance (Average Precision) of the different algorithmic approaches developed against the baseline model on a realistic driving dataset.

| Architecture / Detector | AP@0.5 | AP@0.7 | Main Strategy |
|-------------------------|--------|--------|---------------|
| **Optimized MSER (Our Final Model)** | **83.40%** | **81.90%** | **MSER + Strict HSV Filtering + Asymmetric Padding (-2 penalty)** |
| **Baseline** | 68.70% | 60.90% | Baseline evaluation model |
| **Color-First (detector_alt)** | 27.61% | 8.65% | Pure blue HSV segmentation → local Hough validation |
| **Hybrid (detector_hybrid)** | 24.02% | 10.10% | Blue HSV blobs → geometric refinement with local Hough |
| **Hough Primary** | 1.10% | 0.06% | Global HoughLinesP → H×V intersections |

---

## 1. The Winning Model: Optimized MSER (`detector_mser.py`)

**Performance:** AP@0.5 = 83.40% | AP@0.7 = 81.90%

**Architectural Pipeline:**
1. **Maximally Stable Extremal Regions (MSER)** extraction in grayscale to isolate high-contrast areas (white text on a dark background).
2. Strict geometric filtering (area, width, height, and aspect ratio [0.6 - 4.0]).
3. **Asymmetric Padding:** Algorithmically calculated bounding box expansion (4.5%) to ensure the capture of the outer white border.
4. **Topological HSV Validation:** Mathematical correlation against an ideal 80x40 mask. 

**The Secret to Success (Why it outperforms the Baseline):**
The extraordinary performance of this model lies in the **Active Penalty Matrix**. Instead of using a standard binary correlation (1 and 0), we designed an ideal mask where the outer border has a value of `-2.0`. This means the algorithm severely punishes over-segmentation: if the model attempts to classify a patch of sky or a car reflection, the presence of blue on the outer edges of the matrix tanks its score, discarding false positives with near-perfect efficiency.

---

## 2. Experimental Approach A: Hybrid (Color + Hough)

**Performance:** AP@0.5 = 24.02% | AP@0.7 = 10.10%

**Strategy:**
Use color segmentation (HSV) to find the Region of Interest (ROI) and then execute the probabilistic Hough Transform (HoughLinesP) locally to snap the box exactly to the white borders of the panel.

**Trade-off Interpretation:**
Although this approach improves strict precision (AP@0.7 increases from 8.65% to 10.10% compared to the Color-First model), it loses to the MSER model because the white border of the panels is often degraded by sun glare or distance. This degradation causes Hough to fail in finding structural lines, which misaligns the final bounding box.

**Visual Debug (Case Study `00006.png`):**
* Detected Blob: `(636,54)-(932,333)`
* Expanded search zone: `±35px`
* Hough Lines: `47 H, 17 V` (local to the zone)
* Final adjustment after refinement: `Δx1=-10, Δy1=-6, Δx2=+8, Δy2=+8` (The panel is well framed, but the computational cost and dependence on perfect borders penalize global recall).

---

## 3. Experimental Approach B: Color-First (`detector_alt.py`)

**Performance:** AP@0.5 = 27.61% | AP@0.7 = 8.65%

**Strategy:**
Rely purely on an HSV mask of highly saturated blue (S: 200-255) processed with morphological `OPEN(3x3)` and `CLOSE(7x7)` operations. Hough is used only to confirm that there are straight lines inside the region, not to modify coordinates.

**Conclusion:**
The blue mask is highly specific but generates imprecise boxes (low AP@0.7). By expanding the box in a conservative and fixed manner, it fails to adapt to the real perspective of the panel relative to the camera.

---

## 4. Experimental Approach C: Hough Primary (Why it fails)

**Performance:** AP@0.5 = 1.10% | AP@0.7 = 0.06%

**Strategy:**
Execute HoughLinesP on the global image, classify lines into horizontals and verticals, cross their intersections to form rectangles, and validate with a blue HSV filter.

**Forensic Failure Analysis (Debug `00006.png`):**
* Canny edges: `50,204` edge pixels.
* HoughLinesP raw: `522` lines detected across the entire image.
* Classification: `421` Horizontals, `47` Verticals. (H/V Ratio = 9:1).
* **The Problem:** The image is heavily dominated by the lines of the road, guardrails, and the horizon. Global Hough lines intersect the panels in half, generating `115` erroneous rectangles.
* **Lesson learned:** The Hough transform is highly ineffective as a global searcher in driving environments due to the massive amount of structural background noise.

---

## General Conclusion

Experimentation with multiple architectures empirically demonstrates that color (HSV) and geometric lines (Hough) are weak features when used as primary detectors in real driving scenarios. 

The resounding success of our **MSER Model (83.4% AP)** confirms that regional contrast stability, combined with an asymmetric Padding mathematical design and a topological negative penalty matrix, is the superior approach to solving this problem using classical Computer Vision techniques.