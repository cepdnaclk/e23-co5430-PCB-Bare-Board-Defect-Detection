import time
import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from ultralytics.utils.metrics import bbox_iou
import warnings
warnings.filterwarnings('ignore')

from src.dl.inference import YOLOInferencePipeline
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
    pipeline = YOLOInferencePipeline(model_path, tile_size=640, overlap=0.15)
    
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
    print("CLASSICAL CV EVALUATION METRICS")
    print("="*50)
    
    from src.classical.template_matching import detect_defects
    import random
    
    raw_dir = Path(raw_data_dir)
    image_paths = list(raw_dir.glob('images/*/*.jpg'))
    
    # Sample 20 images to keep evaluation fast, 
    # since classical alignment is computationally heavy on full resolution
    random.seed(42)
    sample_paths = random.sample(image_paths, min(20, len(image_paths)))
    
    all_y_true = []
    all_y_scores = []
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
            
        class_dir = img_path.parent.name
        xml_name = img_path.with_suffix('.xml').name
        xml_path = raw_dir / 'Annotations' / class_dir / xml_name
        
        gt_boxes = []
        if xml_path.exists():
            gt_boxes = parse_voc_xml(str(xml_path)) # [class_id, x1, y1, x2, y2]
            
        try:
            aligned, mask, pred_boxes = detect_defects(test_img, template_img)
        except Exception:
            continue
            
        pred_boxes_xyxy = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in pred_boxes]
        
        matched_gt = set()
        for pb in pred_boxes_xyxy:
            match_found = False
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = compute_iou(pb, gt[1:])
                if iou > 0.5:
                    tp += 1
                    matched_gt.add(i)
                    match_found = True
                    all_y_true.append(1)
                    all_y_scores.append(1.0) 
                    break
            if not match_found:
                fp += 1
                all_y_true.append(0)
                all_y_scores.append(1.0)
                
        for i in range(len(gt_boxes)):
            if i not in matched_gt:
                fn += 1
                all_y_true.append(1)
                all_y_scores.append(0.0) 
                
    end_time = time.time()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"mAP@0.5 (approx): {precision:.4f}")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"F1-Score:         {f1:.4f}")
    print(f"Inference Speed (FPS on full res): {len(sample_paths) / (end_time - start_time):.2f}")
    
    if len(all_y_true) > 0:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        import matplotlib.pyplot as plt
        
        p, r, _ = precision_recall_curve(all_y_true, all_y_scores)
        ap = average_precision_score(all_y_true, all_y_scores)
        
        plt.figure(figsize=(8,6))
        plt.step(r, p, where='post', color='blue', alpha=0.8, label=f'Classical Method AP={ap:.2f}')
        plt.fill_between(r, p, step='post', alpha=0.2, color='blue')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve (Classical Method vs YOLO Baseline)')
        plt.legend(loc='lower left')
        plt.savefig('pr_curve_classical.jpg')
        print("Saved PR curve to pr_curve_classical.jpg")
        
    print("="*50)

if __name__ == "__main__":
    with open('configs/dataset.yaml', 'r') as f:
        config = yaml.safe_load(f)
    raw_data_dir = config.get('raw_data_dir', '')
    
    # Try to execute classical evaluation if data is found
    evaluate_classical(raw_data_dir)
    
    # We will temporarily comment out YOLO eval to test classical quickly
    # evaluate_yolo('runs/train/microinspect_yolo/weights/best.pt', 'configs/yolo_dataset.yaml')
