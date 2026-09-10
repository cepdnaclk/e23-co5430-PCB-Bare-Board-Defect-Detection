"""
Automated Ablation Study & Benchmarking Script
This script runs the Classical Topological Pipeline and the YOLO Deep Learning Pipeline 
on the same test images, measures inference speed, calculates Precision/Recall/F1 via IoU,
generates detailed Confusion Matrices, and creates side-by-side visualization images.
"""

import cv2
import time
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / 'src'))

from classical.template_matching_topological import detect_defects_topological
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
            # Predicted as Background because it missed it completely
            y_pred.append(BACKGROUND_CLASS)
            
    return y_true, y_pred

def run_benchmark():
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
    template_dir = pku_root / 'PCB_USED'
    model_weights = project_root / 'runs' / 'deep_learning' / 'microinspect_v2_medium' / 'weights' / 'best.pt'
    output_dir = project_root / 'outputs' / 'ablation'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We will test on a subset to keep execution time reasonable
    image_paths = list(test_img_dir.glob('*.jpg'))[:10]
    
    tiler = ImageTiler()
    
    metrics = {
        'classical': {'time': [], 'y_true': [], 'y_pred': []},
        'yolo': {'time': [], 'y_true': [], 'y_pred': []}
    }
    
    logging.info(f"Starting Benchmark on {len(image_paths)} images...")
    
    for i, img_path in enumerate(image_paths):
        logging.info(f"Processing {i+1}/{len(image_paths)}: {img_path.name}")
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        
        # Load Template
        template_name = img_path.name.split('_')[0] + '.JPG'
        template_path = template_dir / template_name
        template_img = cv2.imread(str(template_path))
        
        # Load Ground Truth
        txt_path = test_label_dir / img_path.with_suffix('.txt').name
        raw_gt = tiler.parse_yolo_txt(txt_path, w, h)
        gt_boxes = [[gt[1], gt[2], gt[3], gt[4], gt[0]] for gt in raw_gt]
        
        # 1. Classical Method
        start_time = time.time()
        _, _, classical_raw_preds = detect_defects_topological(img, template_img)
        classical_time = time.time() - start_time
        metrics['classical']['time'].append(classical_time)
        
        # Convert [x, y, w, h, class_id] to [x1, y1, x2, y2, class_id]
        classical_preds = [[p[0], p[1], p[0]+p[2], p[1]+p[3], p[4]] for p in classical_raw_preds]
        
        y_true_c, y_pred_c = evaluate_predictions_with_classes(classical_preds, gt_boxes)
        metrics['classical']['y_true'].extend(y_true_c)
        metrics['classical']['y_pred'].extend(y_pred_c)
        
        # 2. YOLO Deep Learning Method
        start_time = time.time()
        yolo_raw_preds = predict_large_image(img_path, model_weights, output_dir=output_dir / 'yolo_temp')
        if yolo_raw_preds is None:
            yolo_raw_preds = []
        yolo_time = time.time() - start_time
        metrics['yolo']['time'].append(yolo_time)
        
        yolo_preds = [[p[0], p[1], p[2], p[3], p[5]] for p in yolo_raw_preds]
        
        y_true_y, y_pred_y = evaluate_predictions_with_classes(yolo_preds, gt_boxes)
        metrics['yolo']['y_true'].extend(y_true_y)
        metrics['yolo']['y_pred'].extend(y_pred_y)
        
        # 3. Save side-by-side visualization (for the first 5 images)
        if i < 10:
            img_c = img.copy()
            img_y = img.copy()
            
            # Draw Ground Truth in Green
            for gt in gt_boxes:
                cv2.rectangle(img_c, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0), 4)
                cv2.rectangle(img_y, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0), 4)
                
            # Draw Classical in Blue
            for p in classical_preds:
                cv2.rectangle(img_c, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 3)
                
            # Draw YOLO in Red
            for p in yolo_preds:
                cv2.rectangle(img_y, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 0, 255), 3)
                
            # Resize for viewing
            h_new = 800
            w_new = int((h_new / h) * w)
            img_c = cv2.resize(img_c, (w_new, h_new))
            img_y = cv2.resize(img_y, (w_new, h_new))
            
            # Add text
            cv2.putText(img_c, "Classical Method (GT=Green, Pred=Blue)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img_y, "YOLO Method (GT=Green, Pred=Red)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            side_by_side = cv2.hconcat([img_c, img_y])
            cv2.imwrite(str(output_dir / f"comparison_{i+1}.jpg"), side_by_side)
            
    # Function to generate Confusion Matrix and calculate Global TP/FP/FN
    def plot_confusion_matrix(y_true, y_pred, title, filename):
        # 0-5 are defects, 6 is Background
        cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))
        
        # Normalize by row (True classes) to get percentages
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm) # Replace NaNs with 0
        
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                         xticklabels=CLASS_NAMES + ['Background'], 
                         yticklabels=CLASS_NAMES + ['Background'])
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(str(output_dir / filename))
        plt.close()
        
        # Calculate Global TP, FP, FN
        # TP: Correctly identified the exact defect class
        tp = np.sum(np.diag(cm)[:6])
        
        # FN: Real defects that were completely missed (Predicted as Background)
        fn_missed = np.sum(cm[:6, 6])
        
        # FP: Background noise mistakenly flagged as a defect
        fp_ghost = np.sum(cm[6, :6])
        
        # Misclassifications: Predicted it was a defect, but guessed the WRONG defect class
        misclass = np.sum(cm[:6, :6]) - tp
        
        # For a binary "Defect vs No-Defect" global metric:
        # We consider a misclassification as BOTH a False Negative (missed the real one) 
        # AND a False Positive (hallucinated the wrong one)
        fp_global = fp_ghost + misclass
        fn_global = fn_missed + misclass
        
        return tp, fp_global, fn_global

    # Generate the CMs and extract global metrics
    c_tp, c_fp, c_fn = plot_confusion_matrix(metrics['classical']['y_true'], metrics['classical']['y_pred'], 'Classical Topological - Confusion Matrix', 'classical_cm.png')
    y_tp, y_fp, y_fn = plot_confusion_matrix(metrics['yolo']['y_true'], metrics['yolo']['y_pred'], 'YOLOv11 - Confusion Matrix', 'yolo_cm.png')

    # Calculate Final Metrics
    def calc_stats(tp, fp, fn, times):
        p = tp / max(1, (tp + fp))
        r = tp / max(1, (tp + fn))
        f1 = 2 * (p * r) / max(0.001, (p + r))
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        return p, r, f1, fps
        
    c_p, c_r, c_f1, c_fps = calc_stats(c_tp, c_fp, c_fn, metrics['classical']['time'])
    y_p, y_r, y_f1, y_fps = calc_stats(y_tp, y_fp, y_fn, metrics['yolo']['time'])
    
    # Generate Graphs
    labels = ['Classical Topological', 'YOLO Deep Learning']
    
    # 1. FPS Graph
    plt.figure(figsize=(8, 6))
    plt.bar(labels, [c_fps, y_fps], color=['blue', 'red'])
    plt.title('Inference Speed Comparison')
    plt.ylabel('Frames Per Second (FPS)')
    plt.savefig(str(output_dir / 'speed_comparison.png'))
    plt.close()
    
    # 2. Accuracy Graph
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.25
    plt.bar(x - width, [c_p, y_p], width, label='Precision', color='royalblue')
    plt.bar(x, [c_r, y_r], width, label='Recall', color='forestgreen')
    plt.bar(x + width, [c_f1, y_f1], width, label='F1-Score', color='darkorange')
    plt.xticks(x, labels)
    plt.title('Accuracy Metrics Comparison')
    plt.ylabel('Score (0-1)')
    plt.legend()
    plt.savefig(str(output_dir / 'accuracy_comparison.png'))
    plt.close()
    
    # Generate Markdown Table
    md_content = f"""# Ablation Study Results: Classical vs Deep Learning

| Metric | Classical Topological | YOLOv11 Medium |
|---|---|---|
| **Precision** | {c_p:.4f} | {y_p:.4f} |
| **Recall** | {c_r:.4f} | {y_r:.4f} |
| **F1-Score** | {c_f1:.4f} | {y_f1:.4f} |
| **Average Speed** | {c_fps:.2f} FPS | {y_fps:.2f} FPS |
| **True Positives** | {c_tp} | {y_tp} |
| **False Positives** | {c_fp} | {y_fp} |
| **False Negatives** | {c_fn} | {y_fn} |

> Note: Precision and Recall were calculated using an IoU threshold of 0.50. Misclassifications are counted as both a False Positive and False Negative.
"""
    with open(project_root / 'docs' / 'ablation_results.md', 'w') as f:
        f.write(md_content)
        
    logging.info(f"Benchmarking Complete! Check the '{output_dir}' and 'docs/' folders for results.")

if __name__ == '__main__':
    run_benchmark()
