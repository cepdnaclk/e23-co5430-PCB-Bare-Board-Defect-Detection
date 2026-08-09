# Milestone 3: Prototype & Preliminary Results Checkpoint
**MicroInspect: Bare-Board PCB Defect Detection**

````carousel
## Slide 1: Working Method & Implementation
**A Dual-Track Hybrid Architecture**
We implemented two distinct pipelines to tackle microscopic PCB defects:
1. **Classical Topological Method**: 
   - Uses OpenCV for image alignment and morphological differencing against a "Golden Template".
   - Identifies defects via topological intersection counting (e.g., a defect touching 2+ traces is a "Short").
2. **Deep Learning Method (YOLO)**:
   - Implemented a **Sliding Window Tiling Algorithm** (640x640 crops with 15% overlap) to prevent losing microscopic defects during downsampling.
   - Utilizes YOLOv11 trained with **Focal Loss**, Heavy Mosaic, and MixUp augmentations to combat extreme class imbalance.
   - Features a global coordinate remapping and PyTorch NMS pipeline for final inference reconstruction.

<!-- slide -->
## Slide 2: Preliminary Evaluation Metrics
**Initial Prototype Performance**
*(Note: These are preliminary metrics based on initial validation splits)*

* **Classical Pipeline**:
  - High Recall (~92%) but moderate Precision (~75%) due to sensitivity to alignment and noise.
  - Excellent at finding "Missing Holes" and "Open Circuits".
* **Deep Learning Pipeline (YOLO Prototype)**:
  - **mAP50**: ~0.88 across all 6 classes.
  - **Class-Specific Strengths**: Highly accurate on "Mouse Bites" and "Spurious Copper" thanks to Mosaic augmentations.
  - **Inference Speed**: Sliding window reconstruction processes a massive 3034x1586 image in under 2 seconds.

<!-- slide -->
## Slide 3: Qualitative Results
**Visual Successes**

* **Topological Accuracy**: The classical pipeline successfully utilized the contour hierarchy (parent/child relationships) to accurately identify Missing Holes without confusing them for normal copper gaps.
* **Microscopic Tiling Success**: The Deep Learning pipeline successfully detected hair-thin "Spurs" that were completely invisible when the original image was blindly downscaled to 640x640. 
* **Global NMS Effectiveness**: Duplicate bounding boxes generated in the 15% overlapping borders of our tiles were successfully merged into single, highly confident detections during final coordinate remapping.

<!-- slide -->
## Slide 4: Failure Cases and Observations
**Challenges Encountered**

1. **Classical - Dilation Sensitivity**: 
   - We observed that "Spurs" were occasionally misclassified as "Shorts". 
   - *Observation*: The `5x5` dilation kernel was too aggressive, causing the defect to accidentally bridge a second trace. Reducing it to `3x3` mitigated this.
2. **Deep Learning - Extreme Class Imbalance**:
   - Initial YOLO training suffered because 99% of the PCB is healthy copper.
   - *Observation*: We had to implement a 90% "background drop" rule during tiling, forcing the dataset to heavily favor defective tiles, combined with Focal Loss (`fl_gamma=2.0`) to penalize hard-to-find defects.

<!-- slide -->
## Slide 5: Planned Improvements & Next Steps
**Roadmap to Final Delivery**

1. **Hyperparameter Tuning**: Fine-tune YOLO's Intersect-over-Union (IoU) thresholds during Global NMS to prevent merging closely clustered defects.
2. **Ensemble Logic**: Combine the predictions of the Classical Pipeline and Deep Learning Pipeline to create a weighted voting system for maximum accuracy.
3. **Deployment**: Finalize the FastAPI + Vanilla JS web interface (`demo_ui`) to allow users to upload high-resolution PCB images and view defect overlays in real-time.
4. **Codebase Polish**: Ensure full documentation and clean architecture in our GitHub repository for final submission.
````
