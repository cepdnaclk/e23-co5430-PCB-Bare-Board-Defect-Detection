"""
Data Preparation & Tiling Script for MicroInspect Deep Learning Pipeline.
This module processes high-resolution PCB images, slicing them into overlapping tiles,
and mathematically translating the YOLO/VOC bounding boxes to the local tile coordinates.
"""

import os
import cv2
import yaml
import logging
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageTiler:
    """
    Object-Oriented Image Tiler for High-Resolution Object Detection datasets.
    Handles slicing images into tiles, tracking overlapping seams, and mapping 
    bounding box coordinates to local patch space.
    """
    
    CLASS_MAP = {
        'Missing_hole': 0,
        'Mouse_bite': 1,
        'Open_circuit': 2,
        'Short': 3,
        'Spur': 4,
        'Spurious_copper': 5
    }

    def __init__(self, tile_size: int = 640, overlap: float = 0.15, background_ratio: float = 0.1):
        """
        Initializes the ImageTiler.
        
        Args:
            tile_size (int): The width and height of the square tiles (e.g., 640).
            overlap (float): The percentage of overlap between adjacent tiles (e.g., 0.15 for 15%).
            background_ratio (float): The probability of keeping a tile that contains no defects.
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.background_ratio = background_ratio
        self.stride = int(tile_size * (1.0 - overlap))
        
    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Computes the Intersection over Box1 Area (IoB1) to determine how much of a defect
        bounding box falls inside the current tile patch.
        
        Args:
            box1: [xmin, ymin, xmax, ymax] - The defect bounding box in global coordinates.
            box2: [xmin, ymin, xmax, ymax] - The tile patch bounding box in global coordinates.
            
        Returns:
            float: The ratio of the intersection area to the total area of box1.
        """
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        
        if box1_area == 0:
            return 0.0
            
        return intersection_area / box1_area

    def parse_yolo_txt(self, txt_path: Path, img_w: int, img_h: int) -> List[List[int]]:
        """
        Parses a YOLO txt file to extract bounding boxes and converts them back to absolute pixel coordinates.
        
        Args:
            txt_path (Path): Path to the YOLO txt annotation file.
            img_w (int): Width of the original image.
            img_h (int): Height of the original image.
            
        Returns:
            List[List[int]]: A list of bounding boxes in format [class_id, xmin, ymin, xmax, ymax].
        """
        bboxes = []
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h_bbox = float(parts[4])
                
                # Convert from normalized to absolute coordinates
                abs_cx = cx * img_w
                abs_cy = cy * img_h
                abs_w = w * img_w
                abs_h = h_bbox * img_h
                
                xmin = int(abs_cx - (abs_w / 2.0))
                ymin = int(abs_cy - (abs_h / 2.0))
                xmax = int(abs_cx + (abs_w / 2.0))
                ymax = int(abs_cy + (abs_h / 2.0))
                
                # Clamp to image boundaries
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(img_w, xmax), min(img_h, ymax)
                
                bboxes.append([class_id, xmin, ymin, xmax, ymax])
        except Exception as e:
            logging.error(f"Error parsing YOLO txt file {txt_path}: {e}")
            
        return bboxes

    def process_image(self, image_path: Path, txt_path: Path, out_img_dir: Path, out_label_dir: Path):
        """
        Reads a high-resolution image, tiles it, translates bounding boxes, and saves the output.
        
        Args:
            image_path (Path): Path to the input image.
            txt_path (Path): Path to the corresponding YOLO txt annotation file.
            out_img_dir (Path): Output directory for saved tile images.
            out_label_dir (Path): Output directory for saved YOLO label text files.
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logging.warning(f"Could not read image: {image_path}")
                return
                
            h, w, _ = img.shape
            bboxes = self.parse_yolo_txt(txt_path, w, h)
            img_name = image_path.stem
            
            for y in range(0, h, self.stride):
                for x in range(0, w, self.stride):
                    # Define tile boundaries
                    x1, y1 = x, y
                    x2, y2 = min(x + self.tile_size, w), min(y + self.tile_size, h)
                    
                    # Pad/shift if the remaining patch is smaller than tile_size
                    if x2 - x1 < self.tile_size:
                        x1 = max(0, w - self.tile_size)
                        x2 = w
                    if y2 - y1 < self.tile_size:
                        y1 = max(0, h - self.tile_size)
                        y2 = h
                        
                    patch = img[y1:y2, x1:x2]
                    patch_box = [x1, y1, x2, y2]
                    patch_bboxes = []
                    
                    # Translate bounding boxes to local tile coordinates
                    for bbox in bboxes:
                        class_id, bx1, by1, bx2, by2 = bbox
                        box_area = [bx1, by1, bx2, by2]
                        overlap_ratio = self._compute_iou(box_area, patch_box)
                        
                        # Keep if at least 10% of the defect is inside the patch
                        if overlap_ratio > 0.1:
                            # 1. Clip global bounding box coordinates to the patch boundaries
                            nx1 = max(bx1, x1)
                            ny1 = max(by1, y1)
                            nx2 = min(bx2, x2)
                            ny2 = min(by2, y2)
                            
                            # 2. Shift to local coordinate space (relative to top-left of patch)
                            local_x1 = nx1 - x1
                            local_y1 = ny1 - y1
                            local_x2 = nx2 - x1
                            local_y2 = ny2 - y1
                            
                            # 3. Convert to YOLO normalized format (cx, cy, width, height)
                            cx = (local_x1 + local_x2) / 2.0 / self.tile_size
                            cy = (local_y1 + local_y2) / 2.0 / self.tile_size
                            bw = (local_x2 - local_x1) / self.tile_size
                            bh = (local_y2 - local_y1) / self.tile_size
                            
                            # Clamp values to [0.0, 1.0] for safety
                            cx, cy = max(0.0, min(cx, 1.0)), max(0.0, min(cy, 1.0))
                            bw, bh = max(0.0, min(bw, 1.0)), max(0.0, min(bh, 1.0))
                            
                            if bw > 0 and bh > 0:
                                patch_bboxes.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                                
                    # Decide whether to save the patch (prevent flooding with pure background)
                    is_empty = len(patch_bboxes) == 0
                    if is_empty and random.random() > self.background_ratio:
                        continue
                        
                    # Save patch and YOLO label file
                    patch_filename = f"{img_name}_{x1}_{y1}.jpg"
                    label_filename = f"{img_name}_{x1}_{y1}.txt"
                    
                    cv2.imwrite(str(out_img_dir / patch_filename), patch)
                    
                    with open(out_label_dir / label_filename, 'w') as f:
                        f.write("\n".join(patch_bboxes))
                        
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")


def main():
    """
    Main entry point for data preparation.
    Reads dataset paths from configuration and orchestrates the tiling process.
    """
    # Navigate to project root to find config (since we are running from src/deep_learning)
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / 'configs' / 'dataset.yaml'
    
    if not config_path.exists():
        logging.error(f"Configuration file not found at {config_path}")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    raw_dir = Path(config.get('dl_data_dir', ''))
    if not raw_dir.exists():
        logging.error(f"Raw data directory does not exist: {raw_dir}")
        return
        
    # Output structured for YOLO
    tiled_dataset_dir = Path(config.get('processed_data_dir', '/home/dilith_s_b_s/UoP/Sem_4/CO5430/CVProject/data/processed'))
    
    tile_size = config.get('tile_size', 640)
    overlap = config.get('overlap', 0.15)
    
    tiler = ImageTiler(tile_size=tile_size, overlap=overlap, background_ratio=0.1)
    
    # Create required YOLO directory structure
    # The dataset is already split into train, valid, test. We maintain this split.
    splits = ['train', 'valid', 'test']
    for split in splits:
        (tiled_dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (tiled_dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
    logging.info(f"Initialized output directories at {tiled_dataset_dir}")
    
    # Iterate over the already existing splits
    for split_name in splits:
        split_img_dir = raw_dir / split_name / 'images'
        split_label_dir = raw_dir / split_name / 'labels'
        
        if not split_img_dir.exists():
            logging.warning(f"Split directory not found: {split_img_dir}")
            continue
            
        image_paths = list(split_img_dir.glob('*.jpg'))
        
        if not image_paths:
            logging.warning(f"No images found in {split_img_dir}")
            continue
            
        logging.info(f"Processing {split_name} split ({len(image_paths)} images)...")
        out_img_dir = tiled_dataset_dir / split_name / 'images'
        out_label_dir = tiled_dataset_dir / split_name / 'labels'
        
        for img_path in tqdm(image_paths, desc=split_name):
            txt_name = img_path.with_suffix('.txt').name
            txt_path = split_label_dir / txt_name
            
            if txt_path.exists():
                tiler.process_image(img_path, txt_path, out_img_dir, out_label_dir)
            else:
                logging.warning(f"YOLO txt annotation not found: {txt_path}")
                
    logging.info(f"Tiling complete! Dataset is saved to {tiled_dataset_dir}")

if __name__ == '__main__':
    main()
