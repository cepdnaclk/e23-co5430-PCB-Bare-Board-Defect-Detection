# MicroInspect: Automated Bare-Board PCB Defect Detection System

This document outlines the architecture and implementation plan for the MicroInspect project, aiming to compare classical computer vision techniques with modern deep learning for detecting microscopic defects on bare-board PCBs.

## User Review Required

> [!IMPORTANT]
> **Repository Location:** The project structure has been constructed *inside* the `e23-co5430-PCB-Bare-Board-Defect-Detection` repository to ensure it's tracked properly by Git.

> [!WARNING]
> **Kaggle Authentication:** To download the `akhatova/pcb-defects` dataset automatically, you will need to have your `kaggle.json` credentials configured. The dataset will be heavily tiled due to extreme class imbalance between the background and microscopic defects.

> [!NOTE]
> **Deep Learning Model:** We will use **YOLOv11** via the `ultralytics` package for the Deep Learning track, as outlined in the proposal. It will be trained in a cloud GPU environment (e.g. Google Colab/Kaggle) using advanced Albumentations augmentations.

## Open Questions

1. **Defect Classification in Baseline:** The classical baseline detects anomalies but currently does not classify them (Mouse bite, Open circuit, etc.). Do we want to implement a heuristic classification step (e.g., based on shape or location) to make it directly comparable to YOLO's multi-class output?
2. **Evaluation Dataset:** Should we set aside a specific subset of the dataset exclusively for the final comparative evaluation (mAP, PR curves) between both methods?

## Proposed Architecture & Structure

```text
e23-co5430-PCB-Bare-Board-Defect-Detection/
├── configs/                  # YAML config files for hyperparameters and paths
├── data/                     # Raw and processed datasets (ignored in git)
├── src/                      # Main source code modules
│   ├── data/                 # Data downloading, tiling, and dataloaders
│   ├── classical/            # ORB homography, CLAHE, and morphological operations
│   ├── dl/                   # Deep learning model definition, training, inference
│   ├── evaluation/           # mAP, Recall, PR curves, and FPS metrics calculation
│   └── utils/                # Visualization, file I/O, coordinate remapping
├── scripts/                  # Executable scripts
│   ├── train_dl.py
│   ├── evaluate.py
│   └── demo.py               # Live demo script
└── README.md                 # Project overview and instructions
```

## Implementation Phases

### Phase 1: Environment Setup & Data Pipeline
- Create `requirements.txt` with OpenCV, PyTorch, Ultralytics, Pandas, Albumentations, Scikit-learn, etc.
- Implement the **tiling strategy**: Slice high-resolution PCB images into smaller patches with overlap and correctly adjust YOLO bounding box annotations for each patch to mitigate class imbalance.

### Phase 2: Track 1 - Baseline Method (Classical CV) - **[CORE COMPLETED]**
- [x] **Image Preprocessing:** Implement **Contrast Limited Adaptive Histogram Equalization (CLAHE)** to mitigate lighting variations before alignment.
- [x] **Feature Extraction & Registration:** Use **ORB** (Oriented FAST and Rotated BRIEF) to detect keypoints on a defect-free template and the test image. Compute Homography using **RANSAC** and warp the test image.
- [x] **Defect Detection:** Perform absolute pixel subtraction between aligned images, followed by morphological opening/closing to filter noise, and extract bounding contours.
- [x] **Evaluation Preparation:** Ensure the baseline pipeline can process a dataset batch to produce bounding boxes for quantitative evaluation.

### Phase 2.1: Baseline Refinements & Open Items (Remaining Work)
Based on codebase analysis, the core of Phase 2 is implemented, but the following critical items remain to make the baseline robust and directly comparable to the DL track:
- **Heuristic Classification:** Address Open Question #1 by classifying defects as "Excess Copper" (Short, Spur, Spurious copper) or "Missing Copper" (Open, Mouse bite, Missing hole). This can be achieved by analyzing the intensity difference sign (`test - template`) within the bounding contour.
- **Adaptive Thresholding:** Replace the hardcoded `threshold=30` in `detect_defects()` with Otsu's Binarization (`cv2.THRESH_OTSU`) for robust noise filtering across varying illumination conditions.
- **Batch Inference Implementation:** Implement the empty `test_classical_pipeline(config_path)` placeholder in `src/classical/template_matching.py` to act as a standalone inference script that loops over the dataset and saves predictions (e.g., YOLO-format `.txt` files), rather than relying purely on the evaluation script for batch processing.

### Phase 3: Track 2 - Improved Method (Deep Learning)
- Set up YOLOv11 training scripts (`scripts/train_dl.py`) designed for cloud GPUs.
- Incorporate the **Albumentations** library for advanced data augmentation (especially **mosaic and copy-paste**) to increase the representation of rare microscopic defects.
- Implement inference pipeline (`src/dl/inference.py`) with tiling and Non-Maximum Suppression (NMS) to eliminate duplicate boxes along tile seams.

### Phase 4: Evaluation & Demonstration
- Implement `scripts/evaluate.py` to calculate **mAP**, **Recall**, and **Inference FPS** for *both* tracks.
- Utilize **Scikit-learn** and **Matplotlib** to generate comparative visualizations such as PR curves and confusion matrices between the two methods.
- Create `scripts/demo.py` to accept a raw test image and visualize the output bounding boxes side-by-side.

## Verification Plan

### Automated Tests
- Verify that coordinate remapping math for tiling is strictly correct.
- Evaluate both methods on the test set and generate comparative graphs.

### Manual Verification
- **Visual Inspection:** Run `scripts/demo.py` on several test images.
- **Metrics Review:** Ensure Recall is adequately high, as missing a defect is worse than a false positive in PCB manufacturing.
