import time
import cv2
import yaml
import torch
import numpy as np
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

from src.dl.inference import get_inference_pipeline
from src.data.tiler import parse_voc_xml, CLASS_MAP

def evaluate_yolo(model_path, dataset_yaml_path):
    # Ultralytics has a built-in validation function which calculates mAP50, mAP50-95, and Recall.
    from ultralytics import YOLO
    
    print(f"Loading model {model_path} for evaluation...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    print(f"Evaluating on {dataset_yaml_path}...")
    metrics = model.val(data=dataset_yaml_path, split='val', imgsz=640, device=0)
    
    print("\n" + "="*50)
    print("DEEP LEARNING (YOLOv11) EVALUATION METRICS")
    print("="*50)
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Recall:       {metrics.box.r.mean():.4f}")  # mean recall across classes
    print(f"Precision:    {metrics.box.p.mean():.4f}")
    
    # Calculate FPS on full size images
    print("\nMeasuring Inference Speed on Full-Res Images...")
    pipeline = get_inference_pipeline(method="single_stage", model_path=model_path, tile_size=640, overlap=0.15)
    
    # We will measure FPS on a dummy 4K image to simulate real PCB sizes
    dummy_img = np.zeros((3000, 4000, 3), dtype=np.uint8)
    
    # Warmup
    pipeline.run_inference(dummy_img)
    
    start_time = time.time()
    num_runs = 10
    for _ in range(num_runs):
        pipeline.run_inference(dummy_img)
    end_time = time.time()
    
    fps = num_runs / (end_time - start_time)
    print(f"Inference Speed (FPS on 4K image): {fps:.2f}")
    print("="*50)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def evaluate_classical(raw_data_dir):
    print("\n" + "="*50)
    print("CLASSICAL CV (TOPOLOGICAL) EVALUATION METRICS")
    print("="*50)
    
    from src.classical.template_matching_topological import detect_defects_topological as detect_defects
    import random
    
    raw_dir = Path(raw_data_dir)
    # Check both the old structure and the new (YOLO-style) structure
    image_paths = list(raw_dir.glob('valid/images/*.jpg'))
    if not image_paths:
        image_paths = list(raw_dir.glob('images/*/*.jpg'))
    
    if not image_paths:
        print(f"  No images found in {raw_dir}. Check 'classical_data_dir' in configs/dataset.yaml.")
        return

    # Sample 20 images to keep evaluation fast,
    # since classical alignment is computationally heavy on full resolution
    random.seed(42)
    sample_paths = random.sample(image_paths, min(20, len(image_paths)))

    tp = fp = fn = 0
    
    print(f"Evaluating {len(sample_paths)} images...")
    start_time = time.time()
    
    for img_path in sample_paths:
        test_img = cv2.imread(str(img_path))
        
        prefix = img_path.name.split('_')[0]
        template_path = raw_dir / 'PCB_USED' / f"{prefix}.JPG"
        template_img = cv2.imread(str(template_path))
        
        if test_img is None or template_img is None:
            continue
            
        gt_boxes = []
        
        # Check for YOLO format labels first (new dataset)
        label_path_yolo = img_path.parent.parent / 'labels' / img_path.with_suffix('.txt').name
        
        # Fallback to old VOC format
        class_dir = img_path.parent.name
        xml_name = img_path.with_suffix('.xml').name
        xml_path = raw_dir / 'Annotations' / class_dir / xml_name
        
        if label_path_yolo.exists():
            height, width = test_img.shape[:2]
            with open(label_path_yolo, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        w = float(parts[3]) * width
                        h = float(parts[4]) * height
                        
                        x1 = x_center - (w / 2)
                        y1 = y_center - (h / 2)
                        x2 = x_center + (w / 2)
                        y2 = y_center + (h / 2)
                        
                        gt_boxes.append([class_id, x1, y1, x2, y2])
        elif xml_path.exists():
            gt_boxes = parse_voc_xml(str(xml_path)) # [class_id, x1, y1, x2, y2]
            
        try:
            aligned, mask, pred_boxes = detect_defects(test_img, template_img)
        except Exception as e:
            print(f"  Warning: inference failed on {img_path.name}: {e}")
            continue
            
        # pred_boxes is now [x, y, w, h, class_id]
        pred_boxes_xyxyc = [[b[0], b[1], b[0]+b[2], b[1]+b[3], b[4]] for b in pred_boxes]
        
        matched_gt = set()
        for pb in pred_boxes_xyxyc:
            match_found = False
            pb_cls = pb[4]
            pb_coords = pb[:4]
            
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                
                gt_cls = gt[0]
                gt_coords = gt[1:]
                iou = compute_iou(pb_coords, gt_coords)
                
                # True Positive requires both IoU > 0.5 AND matching class
                if iou > 0.5 and pb_cls == gt_cls:
                    tp += 1
                    matched_gt.add(i)
                    match_found = True
                    break
            if not match_found:
                fp += 1
                
        for i in range(len(gt_boxes)):
            if i not in matched_gt:
                fn += 1
                
    end_time = time.time()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"F1-Score:         {f1:.4f}")
    print(f"TP: {tp}  FP: {fp}  FN: {fn}")
    print(f"Inference Speed (FPS on full res): {len(sample_paths) / (end_time - start_time):.2f}")

    # The classical pipeline produces binary detections with no confidence scores,
    # so a full PR curve cannot be drawn. We plot the single operating point instead.
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter([recall], [precision], s=200, color='blue', zorder=5,
                label=f'Classical (Topological)  P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision–Recall Operating Point: Classical Topological Method')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.savefig('pr_curve_classical.jpg')
    print("Saved PR operating-point chart to pr_curve_classical.jpg")

    print("="*50)

if __name__ == "__main__":
    with open('configs/dataset.yaml', 'r') as f:
        config = yaml.safe_load(f)
    classical_data_dir = config.get('classical_data_dir', '')
    
    # Try to execute classical evaluation if data is found
    evaluate_classical(classical_data_dir)
    
    # We will temporarily comment out YOLO eval to test classical quickly
    # evaluate_yolo('runs/train/microinspect_yolo/weights/best.pt', 'configs/yolo_dataset.yaml')
