"""
Genetic Algorithm Hyperparameter Tuning Script for MicroInspect.
This script uses YOLO's built-in evolution algorithms to mathematically discover
the absolute perfect learning rates and augmentations for PCB defect detection.
"""

import logging
from pathlib import Path
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def tune_hyperparameters():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_yaml_path = project_root / 'configs' / 'tiled_yolo_dataset.yaml'
    
    if not data_yaml_path.exists():
        logging.error(f"Cannot find dataset config at {data_yaml_path}. Please ensure you have run train.py or prepare_data.py first.")
        return
        
    logging.info("Initializing YOLO model (Medium) for Genetic Algorithm Tuning...")
    model = YOLO('yolo11m.pt')
    
    # Genetic Algorithm Parameters
    # Tuning trains the model from scratch multiple times (iterations). 
    # We use fewer epochs per generation so it doesn't take weeks to complete.
    epochs_per_generation = 30 
    generations = 30 # Number of times to mutate and evolve
    
    logging.info(f"Starting Hyperparameter Evolution...")
    logging.info(f"Generations: {generations}")
    logging.info(f"Epochs per Generation: {epochs_per_generation}")
    logging.info("WARNING: This process is extremely computationally expensive! It will train the model 30 separate times.")
    
    try:
        # Launch the genetic algorithm tuner
        model.tune(
            data=str(data_yaml_path),
            epochs=epochs_per_generation,
            iterations=generations,
            optimizer='auto',
            project=str(project_root / 'runs' / 'deep_learning'),
            name='microinspect_v3_tune',
            batch=8, # Keep it at 8 for the Medium model to prevent memory crashes
            device=0 # Explicitly use the primary GPU
        )
        
        logging.info("Evolution complete!")
        logging.info("The mathematically optimal hyperparameters have been saved as 'best_hyperparameters.yaml' in your runs directory!")
        
    except Exception as e:
        logging.error(f"Hyperparameter evolution failed: {e}")

if __name__ == '__main__':
    tune_hyperparameters()
