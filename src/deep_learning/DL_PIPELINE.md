# MicroInspect: Deep Learning Pipeline

## 1. Introduction
The Deep Learning track of the **MicroInspect** project utilizes the state-of-the-art **YOLO (You Only Look Once)** object detection architecture to identify and classify microscopic manufacturing defects on bare Printed Circuit Boards (PCBs). 

While our Classical Topological method relies on strict mathematical heuristics and image differencing against a "Golden Template", the Deep Learning method learns to recognize the fundamental visual features of the 6 defect classes (Missing hole, Mouse bite, Open circuit, Short, Spur, Spurious copper) directly from the raw data. This allows for rapid region-proposal and highly robust generalization across different PCB designs and lighting conditions.

---

## 2. Milestone 1: Data Preparation & Image Tiling

### The High-Resolution Challenge
The industrial cameras used for PCB inspection capture very high-resolution images (e.g., `3034x1586` pixels). Standard Convolutional Neural Networks (like YOLO) typically expect square input dimensions of around `640x640`. 

If we were to blindly downsample a `3034x1586` image to `640x640`, the scaling algorithm (pixel interpolation) would physically erase hair-thin copper spurs and microscopic pinholes. The neural network cannot detect defects that have been blurred out of existence.

### The Tiling Solution (`prepare_data.py`)
To solve this, we implemented a robust **Sliding Window Tiling Algorithm**. Instead of scaling the image, we crop it into native-resolution chunks.

1. **Grid Cropping**: The algorithm steps across the high-resolution image, cropping it into overlapping `640x640` tiles. We use a **15% overlap (stride)** to ensure that a defect sitting exactly on a cut-line isn't destroyed or split in half.
2. **Coordinate Translation**: The original YOLO `.txt` bounding boxes are provided in global, normalized coordinates. For each tile, the script mathematically computes the sub-pixel intersection (`IoU`) of the global bounding box against the tile boundaries. If a defect falls inside the tile, its coordinates are remapped into the local space of that specific `640x640` crop and saved to a new `.txt` label file.
3. **Class Balancing (Background Dropping)**: A standard PCB is 99% healthy copper and substrate. If we saved every tile, the dataset would be flooded with empty images, causing the model to suffer from extreme class imbalance (learning to predict "no defect" to maximize baseline accuracy). To counter this, the algorithm saves **100% of tiles containing defects**, but uses a digital coin-flip to randomly discard **90% of purely empty background tiles**.
4. **Structured Output**: The pipeline automatically preserves the original `train/valid/test` dataset splits and exports the final tiled images and labels directly into the `/data/processed/` directory, making it instantly ready for YOLO training.

---

## 3. Milestone 2: Model Configuration & Training

### Dynamic Dataset Generation
YOLO requires a strict `data.yaml` file to understand the dataset structure. Instead of hardcoding this, our training script (`train.py`) programmatically generates `configs/tiled_yolo_dataset.yaml`. This ensures the model explicitly trains on the `/data/processed/` **tiled** images, avoiding any confusion with the raw, high-resolution dataset.

### Addressing Microscopic Defect Imbalance
The core challenge in training this model is **Recall** and **Class Imbalance**. The defects are tiny, and even after background dropping, normal copper vastly outnumbers defective copper. To force the neural network to pay attention to these hard-to-see defects, we heavily modified the YOLO hyperparameters:

1. **Focal Loss (`fl_gamma = 2.0`)**: Standard Cross-Entropy loss treats all errors equally. Focal Loss applies a mathematical curve that heavily penalizes the model for getting "hard" predictions wrong (like missing a microscopic spur), while ignoring "easy" predictions.
2. **Heavy Mosaic (`mosaic = 1.0`)**: This augmentation splices 4 random tiles together during training. It forces the model to learn features at different local scales and contexts.
3. **MixUp (`mixup = 0.2`)**: Blends overlapping images together, helping the model generalize against varying board colors, lighting, and textures.
4. **Classification Boost (`cls = 2.0`)**: We scaled up the classification loss. It is visually very difficult to distinguish a "Spur" (extra copper attached to one trace) from a "Short Circuit" (extra copper bridging two traces) in a tiny 640x640 crop. This forces the model to heavily penalize misclassification.

### Execution
The training loop utilizes Early Stopping (`patience=20`) to prevent overfitting. Once completed, the final optimized weights are automatically saved to `/runs/deep_learning/weights/` for use in the final Inference Engine.

---

*(This document will be updated as we proceed to Milestone 3: Inference and Coordinate Remapping)*
