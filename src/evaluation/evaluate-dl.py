"""
Deep Learning (YOLOv11) Dedicated Testing Script
This script specifically tests the trained YOLO model on the PKU test dataset
using sliding-window inference, and generates detailed performance metrics:
- Confusion Matrix
- Precision, Recall, F1-Score
- Accuracy
"""

import cv2
import time
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / 'src'))

from deep_learning.predict import predict_large_image
from deep_learning.prepare_data import ImageTiler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
BACKGROUND_CLASS = 6 # Index 6 represents Background (No Defect)

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / float(max(1, box1_area + box2_area - intersection_area))
    return iou

def evaluate_predictions_with_classes(preds, gt_boxes, iou_thresh=0.5):
    """
    preds: list of [x1, y1, x2, y2, class_id]
    gt_boxes: list of [x1, y1, x2, y2, class_id]
    Returns y_true, y_pred for generating a Confusion Matrix
    """
    y_true = []
    y_pred = []
    
    matched_gt = set()
    
    for pred in preds:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = calculate_iou(pred[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
                
        if best_iou >= iou_thresh:
            matched_gt.add(best_gt_idx)
            y_true.append(int(gt_boxes[best_gt_idx][4]))
            y_pred.append(int(pred[4]))
        else:
            # False Positive (Predicted a defect, but it was just Background)
            y_true.append(BACKGROUND_CLASS)
            y_pred.append(int(pred[4]))
            
    # False Negatives (Missed ground truth boxes)
    for i, gt in enumerate(gt_boxes):
        if i not in matched_gt:
            y_true.append(int(gt[4]))
            y_pred.append(BACKGROUND_CLASS)
            
    return y_true, y_pred

def run_dl_test():
    # Paths — read from the shared config file so this works on any machine
    import yaml
    config_path = project_root / 'configs' / 'dataset.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    pku_root = Path(config.get('dl_data_dir', 'data/raw/PKU-Market-PCB'))
    if not pku_root.is_absolute():
        pku_root = project_root / pku_root
    test_img_dir = pku_root / 'test' / 'images'
    test_label_dir = pku_root / 'test' / 'labels'
    model_weights = project_root / 'runs' / 'deep_learning' / 'microinspect_v2_medium' / 'weights' / 'best.pt'
    output_dir = project_root / 'outputs' / 'dl_evaluation'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(test_img_dir.glob('*.jpg'))
    tiler = ImageTiler()
    
    y_true_all = []
    y_pred_all = []
    inference_times = []
    
    logging.info(f"Starting Deep Learning Testing on {len(image_paths)} images...")
    
    # We use tqdm for a nice progress bar since evaluating all images takes time
    for img_path in tqdm(image_paths, desc="Evaluating Images"):
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        
        # Load Ground Truth
        txt_path = test_label_dir / img_path.with_suffix('.txt').name
        raw_gt = tiler.parse_yolo_txt(txt_path, w, h)
        gt_boxes = [[gt[1], gt[2], gt[3], gt[4], gt[0]] for gt in raw_gt]
        
        # YOLO Deep Learning Inference
        start_time = time.time()
        # Suppress ultralytics logging by sending to temp directory and limiting prints
        yolo_raw_preds = predict_large_image(img_path, model_weights, output_dir=output_dir / 'visualizations')
        if yolo_raw_preds is None:
            yolo_raw_preds = []
        inference_times.append(time.time() - start_time)
        
        yolo_preds = [[p[0], p[1], p[2], p[3], p[5]] for p in yolo_raw_preds]
        
        y_true, y_pred = evaluate_predictions_with_classes(yolo_preds, gt_boxes)
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    logging.info("Generating Performance Metrics...")
    
    # Calculate Sklearn Metrics
    # Filter out True Negatives (Background -> Background) if any somehow snuck in
    accuracy = accuracy_score(y_true_all, y_pred_all)
    
    # Calculate precision, recall, f1 for each class
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=list(range(6)), zero_division=0
    )
    
    # Macro averages (ignoring background as a "predicted defect")
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    
    avg_fps = 1.0 / np.mean(inference_times)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(7)))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                     xticklabels=CLASS_NAMES + ['Background'], 
                     yticklabels=CLASS_NAMES + ['Background'])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('YOLOv11 Deep Learning - Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(str(output_dir / 'dl_confusion_matrix.png'))
    plt.close()
    
    # Generate Markdown Results
    md_content = f"""# Deep Learning (YOLOv11) Performance Evaluation

Tested on {len(image_paths)} high-resolution images from the PKU-Market-PCB dataset using Sliding-Window Inference.

## Global Metrics
| Metric | Score |
|---|---|
| **Macro Precision** | {macro_precision:.4f} |
| **Macro Recall** | {macro_recall:.4f} |
| **Macro F1-Score** | {macro_f1:.4f} |
| **Overall Accuracy** | {accuracy:.4f} |
| **Average Speed** | {avg_fps:.2f} FPS |

## Per-Class Metrics
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
"""
    for i, cls in enumerate(CLASS_NAMES):
        md_content += f"| {cls} | {precision_per_class[i]:.4f} | {recall_per_class[i]:.4f} | {f1_per_class[i]:.4f} |\n"
        
    md_content += "\n\n*(Note: Metrics calculated at IoU threshold = 0.50)*"

    with open(output_dir / 'dl_performance_report.md', 'w') as f:
        f.write(md_content)
        
    logging.info(f"Testing Complete! Results saved to {output_dir}")
    print("\n--- RESULTS ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {macro_precision:.4f}")
    print(f"Recall:    {macro_recall:.4f}")
    print(f"F1-Score:  {macro_f1:.4f}")
    print(f"Speed:     {avg_fps:.2f} FPS")

if __name__ == '__main__':
    run_dl_test()
