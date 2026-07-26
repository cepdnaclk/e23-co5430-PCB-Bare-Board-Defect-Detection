# MicroInspect: Automated Bare-Board PCB Defect Detection System

MicroInspect is a computer vision project designed to detect and classify microscopic manufacturing defects on bare-board Printed Circuit Boards (PCBs). 

The core of this project is a direct comparative study between two distinct approaches:
1. **Classical Computer Vision:** A highly robust, heuristic-based pipeline utilizing ORB feature extraction, RANSAC Homography alignment, Otsu's adaptive thresholding, and morphological image subtraction.
2. **Deep Learning:** A modern pipeline utilizing YOLOv11 trained on heavily augmented and tiled datasets to identify microscopic defects.

Both methods are designed to identify the 6 standard defect classes: `Missing hole`, `Mouse bite`, `Open circuit`, `Short`, `Spur`, and `Spurious copper`.

---

## 📂 Project Structure

```text
e23-co5430-PCB-Bare-Board-Defect-Detection/
├── configs/                  # YAML config files for datasets and hyperparameters
├── demo_ui/                  # FastAPI web interface for live testing
│   ├── app.py                # Main web server script
│   ├── static/               # HTML, CSS, JS frontend files
│   └── temp/                 # Temporary storage for uploaded images
├── scripts/                  # CLI executable scripts
│   ├── demo.py               # Runs inference on a single image via CLI
│   ├── evaluate.py           # Calculates mAP, Recall, and FPS for the pipelines
│   └── train_dl.py           # Deep Learning training script
├── src/                      # Core source code modules
│   ├── classical/            # Template matching, alignment, and heuristic rules
│   ├── dl/                   # YOLO inference pipelines
│   ├── evaluation/           # Metrics calculation utilities
│   └── data/                 # Data downloading and tiling utilities
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 How to Run the Web Interface (Demo UI)

We have built a fully functional web interface that allows you to easily upload PCB images and instantly visualize the defect detection results from either the Deep Learning or Classical pipeline.

### 1. Install Dependencies
Ensure you have all the required Python packages installed:
```bash
pip install -r requirements.txt
```

### 2. Start the Server
Navigate to the root directory of the project and start the FastAPI server using Uvicorn:
```bash
# Make sure you run this from the project root directory!
python -m uvicorn demo_ui.app:app --reload --port 8000
```

### 3. Access the UI
Open your web browser and go to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### 4. Using the Interface
1. **Select Method:** Choose either *Deep Learning* or *Classical* from the dropdown menu.
2. **Upload Test Image:** Upload a defective PCB image you want to inspect.
3. **Upload Template Image (Classical Only):** If you selected the Classical method, you *must* also upload the corresponding defect-free "Golden" template image.
4. **Analyze:** Click the Analyze button. The UI will process the image in the background and display the bounding boxes alongside the confidence scores or class names!

---

## 🛠️ CLI Usage

If you prefer the command line, you can bypass the UI entirely.

**To run the Deep Learning model on a single image:**
```bash
python scripts/demo.py --method dl --test_img path/to/image.jpg
```

**To run the Classical pipeline on a single image:**
```bash
python scripts/demo.py --method classical --test_img path/to/image.jpg --template_img path/to/template.jpg
```

**To run batch evaluation on the datasets:**
```bash
python scripts/evaluate.py
```
