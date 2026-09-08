# MicroInspect: Classical PCB Defect Detection

````carousel
## 1. Dataset Status and Source

- **Source**: `akhatova/pcb-defects` — PCB_DATASET (classical pipeline); `arnablaha05/deep-pcb` — PKU-Market PCB (DL pipeline).
- **Structure**: Composed of matched image pairs — a faultless **template** image and a corresponding **defective test** image shot under similar conditions.
- **Defect Classes (6)**:
  - **Subtractive** (missing copper): Missing hole, Mouse bite, Open circuit
  - **Additive** (excess copper): Short, Spur, Spurious copper
- **Lighting Handling**: CLAHE (Contrast Limited Adaptive Histogram Equalization, `clipLimit=2.0`, `tileGridSize=(8,8)`) is applied to both images before feature extraction to normalise illumination variations.
- **Rotation Handling**: The dataset contains images at arbitrary orientations. A robust **4-way rotation check** (0°, 90°, 180°, 270°) is performed during alignment to handle these cases.

<!-- slide -->
## 2. Preprocessing & Annotation Plan

Before defects can be classified, the pipeline executes a strict preprocessing flow:

1. **CLAHE Normalisation**: Both the test and template images are converted to grayscale and enhanced with CLAHE. This normalises local contrast differences caused by lighting variations — a prerequisite for reliable ORB keypoint detection.

2. **Feature-Based Alignment (ORB + RANSAC Homography)**: ORB (Oriented FAST and Rotated BRIEF) keypoints are extracted from both images. The pipeline **tries all 4 rotations** (0°, 90°, 180°, 270°) of the test image and keeps the rotation that produces the most inlier matches. A homography matrix is computed via RANSAC and `cv2.warpPerspective` registers the test image onto the template coordinate space.

3. **Image Differencing**: `cv2.absdiff` computes a pixel-wise absolute difference between the CLAHE-enhanced aligned test image and the template. Regions that differ represent potential defects.

4. **Morphological Extraction**: Otsu's binarization is applied to the difference map. A `(5,5)` kernel is then used for **morphological opening** (removes salt-and-pepper noise) followed by **morphological closing** (fills small gaps within defect blobs). `cv2.findContours` with `RETR_EXTERNAL` extracts individual defect contours. Contours below 130 px² are discarded as noise.

<!-- slide -->
## 3. Baseline Pipeline & Methodology
*(Based on `template_matching_topological.py`)*

Unlike standard Deep Learning approaches, our pipeline uses pure **Topological Computer Vision**. For each candidate defect contour, the following decision tree is applied:

- **Step 1 — Missing Hole (Contour Hierarchy Check)**: `cv2.RETR_TREE` is used on the **Otsu-binarized template** to find copper rings (contours that contain a child hole). If the defect centroid falls inside such a child contour (tested with `cv2.pointPolygonTest`), it is immediately classified as a **Missing Hole** — no further processing needed.

- **Step 2 — Additive vs Subtractive Branch**: The local mean pixel intensity inside the bounding box is compared between the test and template:
  - `test_mean > template_mean` → **Additive** (excess copper present)
  - `test_mean ≤ template_mean` → **Subtractive** (copper removed)

- **Step 3 — Intersection Counting**: The defect mask is dilated with an `(11,11)` kernel and bitwise-ANDed with a copper mask. `cv2.connectedComponents` counts the number of distinct trace segments the defect touches:
  - **Subtractive** (AND with `test_copper` — remaining trace segments in the defective image):
    - `count < 2` → **Mouse Bite** (edge chip on one trace, or isolated removal)
    - `count ≥ 2` → **Open Circuit** (trace severed into 2+ disconnected pieces)
  - **Additive** (AND with `template_copper` — healthy traces in the template):
    - `count = 0` → **Spurious Copper** (isolated blob, touches no trace)
    - `count = 1` → **Spur** (protrusion attached to one trace)
    - `count ≥ 2` → **Short Circuit** (bridges two or more distinct traces)

<!-- slide -->
## 4. Initial Implementation Results

The transition from basic heuristics (`template_matching.py`) to **Topological Classification** (`template_matching_topological.py`) delivers the following qualitative improvements:

- **Correct Missing Hole Detection**: The contour hierarchy check correctly identifies the copper annular ring around a drilled hole in the template. A defect centroid falling inside that ring is unambiguously classified as a Missing Hole — independent of defect size, circularity, or thresholding artifacts. The previous heuristic had no mechanism for this class at all.

- **Context-Aware Subtractive Classification**: By counting remaining trace segments in the defective image (`test_copper`) after dilation, the classifier correctly distinguishes a Mouse Bite (damage to a single trace edge) from an Open Circuit (trace fully severed) — without any hand-crafted shape rules.

- **Context-Aware Additive Classification**: By counting how many healthy template traces (`template_copper`) the excess copper touches, the classifier correctly distinguishes Spurious Copper (isolated), Spur (one connection), and Short Circuit (two or more connections) — making classification topologically grounded rather than aspect-ratio based.

- **No GPU Required**: The entire pipeline uses OpenCV's optimised C++ backend — bitwise operations, connected components, contour hierarchy. Full-resolution PCB inference is feasible on CPU-only edge hardware.

- **Demo Output**: Running on sample image `01_missing_hole_01.jpeg` via the FastAPI demo UI (`demo_ui/app.py`) produces three output artefacts saved to `outputs/classical_topological/`:
  - `_aligned.jpg` — the homography-registered test image
  - `_mask.jpg` — the binary morphological difference mask showing detected defect regions
  - `_result.jpg` — annotated image with green bounding boxes and topological class labels

<!-- slide -->
## 5. Current Challenges & Next Steps

**Challenges:**

- **Dilation Kernel Sensitivity**: The `(11,11)` dilation kernel is hardcoded. On dense PCB layouts where traces are close together, this kernel size may cause an Additive defect touching only one trace (Spur) to falsely bridge to a neighbouring trace and be misclassified as a Short Circuit. A `(5,5)` alternative is noted in the code comments but not yet adaptive.

- **Silent Alignment Failure**: When ORB fails to find enough inlier matches across all 4 rotation attempts (e.g., on featureless or very uniform PCB regions), the pipeline silently falls back to using the original unaligned image with an identity homography. This guarantees false-positive "edge sliver" detections in the difference mask with no warning raised to the caller.

- **Fixed Noise Filter Threshold**: The 130 px² minimum contour area filter is a hardcoded constant. Very small defects such as hairline open circuits may be silently dropped, while loosening the threshold increases false positives.

**Next Steps:**

- **Adaptive Dilation**: Dynamically compute the dilation kernel size per-defect based on the distance transform of the nearest adjacent copper trace, preventing false Short Circuit classifications on dense layouts.

- **Alignment Failure Logging**: Add an explicit warning when `best_im1_reg is None` so downstream users and the evaluation script can flag unreliable results rather than silently consuming garbage detections.

- **Quantitative Evaluation**: Run `scripts/evaluate.py` on the full `PCB_DATASET` (once available) to obtain Precision, Recall, F1-Score, and FPS for the topological method.

- **YOLO Comparison**: Once the YOLOv11 model weights (`best.pt`) are available from training, uncomment `evaluate_yolo()` in `scripts/evaluate.py` and plot both the YOLO PR curve and the classical operating point on the same axes for a direct side-by-side comparison.
````
