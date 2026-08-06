"""
Inference and Coordinate Remapping Script for MicroInspect Deep Learning Pipeline.
This module executes sliding-window inference on massive high-resolution PCB images,
remaps all predicted local bounding boxes back to the global coordinate space,
and applies Non-Maximum Suppression to filter overlap duplicates.
"""

import cv2
import torch
import torchvision
import numpy as np
import logging
from pathlib import Path
from ultralytics import YOLO
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CLASS_NAMES = {
    0: 'Missing_hole',
    1: 'Mouse_bite',
    2: 'Open_circuit',
    3: 'Short',
    4: 'Spur',
    5: 'Spurious_copper'
}

# Distinct colors for each defect type
COLORS = [
    (0, 0, 255),    # Missing_hole (Red)
    (0, 255, 255),  # Mouse_bite (Yellow)
    (255, 0, 0),    # Open_circuit (Blue)
    (255, 0, 255),  # Short (Magenta)
    (0, 255, 0),    # Spur (Green)
    (255, 255, 0)   # Spurious_copper (Cyan)
]

def predict_large_image(image_path: Path, model_path: Path, output_dir: Path, 
                        tile_size: int = 640, overlap: float = 0.15, 
                        conf_thresh: float = 0.25, iou_thresh: float = 0.45):
    """
    Tiles a high-resolution image, runs YOLO inference on each tile,
    remaps coordinates globally, applies NMS, and saves the final result.
    """
    if not image_path.exists():
        logging.error(f"Image not found: {image_path}")
        return
        
    if not model_path.exists():
        logging.error(f"Model weights not found at {model_path}. Did you finish training?")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading YOLO model from {model_path}")
    model = YOLO(str(model_path))
    
    logging.info(f"Reading high-resolution image: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        logging.error("Failed to load image via OpenCV.")
        return
        
    img_disp = img.copy() # For drawing the final global bounding boxes
    h, w, _ = img.shape
    stride = int(tile_size * (1 - overlap))
    
    global_predictions = []
    
    # 1. SLIDING WINDOW INFERENCE
    logging.info("Starting sliding-window inference and coordinate remapping...")
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Calculate tile bounds
            x1, y1 = x, y
            x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
            
            # Shift window if hitting the right/bottom edge to maintain tile_size
            if x2 - x1 < tile_size:
                x1 = max(0, w - tile_size)
                x2 = w
            if y2 - y1 < tile_size:
                y1 = max(0, h - tile_size)
                y2 = h
                
            tile = img[y1:y2, x1:x2]
            
            # Predict on this 640x640 tile
            results = model(tile, verbose=False, conf=conf_thresh)
            
            # 2. COORDINATE REMAPPING
            for r in results:
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy() # [local_x1, local_y1, local_x2, local_y2]
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    
                    # Map back to the gigantic global image coordinates!
                    g_x1 = b[0] + x1
                    g_y1 = b[1] + y1
                    g_x2 = b[2] + x1
                    g_y2 = b[3] + y1
                    
                    global_predictions.append([g_x1, g_y1, g_x2, g_y2, conf, cls_id])
                    
    if not global_predictions:
        logging.info("No defects found in this image.")
        return
        
    global_predictions = np.array(global_predictions)
    final_boxes = []
    
    # 3. GLOBAL NON-MAXIMUM SUPPRESSION (NMS)
    logging.info("Applying global Non-Maximum Suppression to remove overlap duplicates...")
    
    for cls in range(len(CLASS_NAMES)):
        cls_mask = global_predictions[:, 5] == cls
        cls_preds = global_predictions[cls_mask]
        
        if len(cls_preds) == 0:
            continue
            
        boxes = torch.tensor(cls_preds[:, :4], dtype=torch.float32)
        scores = torch.tensor(cls_preds[:, 4], dtype=torch.float32)
        
        # PyTorch built-in NMS
        keep_indices = torchvision.ops.nms(boxes, scores, iou_thresh)
        
        for idx in keep_indices:
            final_boxes.append(cls_preds[idx])
            
    logging.info(f"Filtered {len(global_predictions)} raw predictions down to {len(final_boxes)} final unique defects.")
    
    # 4. VISUALIZATION & SAVING
    for box in final_boxes:
        x1, y1, x2, y2, conf, cls_id = box
        color = COLORS[int(cls_id) % len(COLORS)]
        label = f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}"
        
        # Draw thick boxes for high-res visibility
        cv2.rectangle(img_disp, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
        
        # Draw label background for readability
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img_disp, (int(x1), int(y1) - 25), (int(x1) + text_w, int(y1)), color, -1)
        cv2.putText(img_disp, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        
    output_path = output_dir / f"{image_path.stem}_inference.jpg"
    cv2.imwrite(str(output_path), img_disp)
    logging.info(f"Saved final high-resolution inference image to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MicroInspect YOLO Sliding Window Inference")
    parser.add_argument('--image', type=str, required=True, help="Path to high-resolution test image")
    parser.add_argument('--weights', type=str, default='runs/deep_learning/microinspect_v1/weights/best.pt', help="Path to trained YOLO weights")
    parser.add_argument('--output', type=str, default='data/inference_results', help="Output directory")
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    img_path = Path(args.image)
    if not img_path.is_absolute():
        img_path = project_root / img_path
        
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = project_root / weights_path
        
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
        
    predict_large_image(img_path, weights_path, out_dir)
