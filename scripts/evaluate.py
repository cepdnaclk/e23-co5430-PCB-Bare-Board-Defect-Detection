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

def evaluate_classical(test_img_dir, template_img_dir):
    print("\nClassical CV Evaluation is conceptually complex due to reliance on template pairs.")
    print("Please use scripts/demo.py to visually verify classical pipeline results.")
    
if __name__ == "__main__":
    evaluate_yolo('runs/train/microinspect_yolo/weights/best.pt', 'configs/yolo_dataset.yaml')
