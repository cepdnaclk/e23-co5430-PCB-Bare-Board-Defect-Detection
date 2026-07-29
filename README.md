# MicroInspect: PCB Bare-Board Defect Detection

MicroInspect is a highly robust, hybrid Computer Vision pipeline designed to automatically detect and classify manufacturing defects on bare Printed Circuit Boards (PCBs). 

This project tackles the challenge using a dual-pronged approach, allowing users to choose between a blazing-fast **Classical Topological Computer Vision** pipeline that operates without needing a GPU and a state-of-the-art **Deep Learning (YOLO)** object detection model.

## Supported Defect Classes
The system detects 6 common types of PCB manufacturing defects, broken down into two main topological categories:

**Subtractive Defects (Missing Copper)**
1. **Missing Hole**: An unplated or completely missing drilled hole inside a copper pad.
2. **Mouse Bite**: A jagged chunk of copper missing from the edge of a trace.
3. **Open Circuit**: A complete severing of a copper trace into two or more pieces.

**Additive Defects (Excess Copper)**
1. **Short Circuit**: Excess copper that incorrectly bridges two distinct traces.
2. **Spur**: A protrusion of excess copper attached to a single trace.
3. **Spurious Copper**: An isolated island of excess copper not touching any traces.


---

## Methodology Pipelines

### 1. Classical Template Matching
Located in `src/classical/template_matching.py`. An extremely fast baseline that uses ORB feature-matching and homographies to perfectly align a defective Test image to a flawless Template image (supporting arbitrary rotations like 90°, 180°, 270°). It extracts defects using absolute image differencing and categorizes them loosely based on intensity heuristics.

### 2. Advanced Classical Topological Classification
Located in `src/classical/template_matching_topological.py`. This method takes the aligned difference masks and applies strict mathematical topological rules to perfectly categorize defects:
- **Missing Hole Detection**: Uses `cv2.RETR_TREE` contour hierarchies to find copper rings in the template image, proving a missing hole exists if the defect falls inside a child contour.
- **Intersection Counting**: Dilates defects and mathematically counts how many distinct healthy traces they intersect to differentiate between Mouse Bites (1 stump), Open Circuits (2+ stumps), Spurs (1 connection), and Shorts (2+ connections).

### 3. Deep Learning Method (YOLO)
Located in `src/dl/inference.py`. Uses a YOLO-based architecture trained on the DeepPCB dataset to directly regress bounding boxes and classify defects in a single pass.

---

## Developer Guide: Running the Demo UI

MicroInspect comes with a beautiful, fully functional Web Interface built with **FastAPI** (Backend) and Vanilla JS/CSS (Frontend). 

### Prerequisites
Make sure your Python environment has the required dependencies installed:
```bash
pip install fastapi uvicorn python-multipart opencv-python-headless numpy
# Note: YOLO/PyTorch dependencies are required if running the 'dl' pipeline.
```

### Starting the Server
1. Open your terminal and navigate to the project root directory:
```bash
cd /path/to/e23-co5430-PCB-Bare-Board-Defect-Detection
```
2. Start the Uvicorn development server:
```bash
python3 -m uvicorn demo_ui.app:app --reload --port 8000
```
3. Open your web browser and navigate to: [http://localhost:8000](http://localhost:8000)

### Using the Interface
1. **Select a Mode**: Choose between `DEEP_LEARNING [YOLO]`, `CLASSICAL [TEMPLATE]`, or `CLASSICAL [TOPOLOGICAL]`.
2. **Upload Images**:
   - **Test Image**: Drag and drop your defective PCB image here (required for all modes).
   - **Template Image**: Drag and drop the "golden" faultless PCB image here (only required if running a Classical mode).
3. **Scan**: Click `INITIATE_SCAN()`. The server will process the image, classify the defects, and return a hacker-themed dashboard displaying the bounding boxes, aligned masks, and classification labels!

---

## CLI Usage
If you prefer running tests from the command line without starting a server, you can use the `demo.py` script:

```bash
# Run Deep Learning
python scripts/demo.py --test_img data/test_img.jpg --method dl

# Run Classical Topological
python scripts/demo.py --test_img data/test_img.jpg --template_img data/template_img.jpg --method classical_topological
```
Results will be automatically generated and saved in the `outputs/` directory.

---