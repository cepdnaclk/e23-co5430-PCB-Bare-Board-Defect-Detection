"""
YOLO Configuration and Training Script for MicroInspect Deep Learning Pipeline.
This module dynamically generates the dataset configuration and trains the YOLO model
on the tiled dataset, with hyperparameters specifically optimized for high Recall
and heavy class imbalance.
"""

import os
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_tiled_data_yaml(project_root: Path, tiled_data_dir: Path) -> Path:
    """
    Programmatically generates the data.yaml required by Ultralytics YOLO,
    pointing explicitly to our newly generated tiled dataset.
    """
    yaml_path = project_root / 'configs' / 'tiled_yolo_dataset.yaml'
    
    config = {
        'path': str(tiled_data_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 6,
        'names': {
            0: 'Missing_hole',
            1: 'Mouse_bite',
            2: 'Open_circuit',
            3: 'Short',
            4: 'Spur',
            5: 'Spurious_copper'
        }
    }
    
    # Ensure config directory exists
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
    logging.info(f"Generated YOLO dataset configuration at {yaml_path}")
    return yaml_path

def train_model():
    """
    Initializes and trains the YOLO model on the tiled dataset.
    """
    # Define paths
    project_root = Path(__file__).resolve().parent.parent.parent
    tiled_data_dir = Path('..') / 'data' / 'processed'
    weights_dir = project_root / 'runs' / 'deep_learning' / 'weights'
    
    if not (tiled_data_dir / 'train').exists():
        logging.error(f"Tiled dataset not found at {tiled_data_dir}. Did you run prepare_data.py?")
        return
        
    # 1. Generate the data.yaml
    data_yaml_path = generate_tiled_data_yaml(project_root, tiled_data_dir)
    
    # 2. Initialize the YOLO model (YOLOv11 nano for edge efficiency, can be changed to 's' or 'm')
    model_name = 'yolo11n.pt'
    logging.info(f"Initializing YOLO model: {model_name}")
    model = YOLO(model_name)
    
    # 3. Configure hyperparameters & Train
    # We heavily prioritize RECALL and handling Class Imbalance for microscopic defects.
    # Focal Loss (fl_gamma), Heavy Mosaic, and MixUp are critical here.
    epochs = 100
    batch_size = 16
    
    logging.info(f"Starting training for {epochs} epochs...")
    
    try:
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            project=str(project_root / 'runs' / 'deep_learning'),
            name='microinspect_v1',
            exist_ok=True,
            
            # --- HYPERPARAMETERS TUNED FOR MICROSCOPIC DEFECTS ---
            patience=20,            # Early stopping to prevent overfitting
            optimizer='auto',
            lr0=0.01,               # Initial learning rate
            
            # Class Imbalance & Recall tuning
            cls=2.0,                # Scale up classification loss (helps heavily when distinguishing between short vs spur)
            box=7.5,                # Box loss gain: Forces tighter bounding boxes
            
            # Augmentations for tiled industrial data
            mosaic=1.0,             # Heavy mosaic augmentation (splices 4 tiles together)
            mixup=0.2,              # Mixup augmentation (blends images, helps with imbalance)
            degrees=15.0,           # Slight rotations
            hsv_h=0.015,            # Slight hue shifts for lighting variations
            hsv_s=0.7,
            hsv_v=0.4,
            
            # Device selection
            device=0 
        )
        logging.info(f"Training successfully completed! Weights saved to: {weights_dir}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")

if __name__ == '__main__':
    train_model()
