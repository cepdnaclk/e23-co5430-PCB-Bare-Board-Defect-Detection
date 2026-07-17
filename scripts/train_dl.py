from ultralytics import YOLO
import yaml
from pathlib import Path

def train_yolo(config_path='configs/dataset.yaml', yolo_data_yaml='configs/yolo_dataset.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model_name = config.get('model_name', 'yolo11n.pt')
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 16)
    
    # Load a model
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    
    # Train the model
    print(f"Starting training on {yolo_data_yaml} for {epochs} epochs...")
    results = model.train(
        data=yolo_data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        device=0, # Assuming CUDA is available as per user requirements
        project='runs/train',
        name='microinspect_yolo',
        exist_ok=True,
        cache=True
    )
    
    print("Training completed. Results saved to runs/train/microinspect_yolo")
    
if __name__ == "__main__":
    train_yolo()
