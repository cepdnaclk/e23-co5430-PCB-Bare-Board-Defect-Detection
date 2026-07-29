import os
# pyrefly: ignore [missing-import]
import cv2
import glob
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import random

# class mapping for different types of defects on PCB
CLASS_MAP = {
    'Missing_hole': 0,
    'Mouse_bite': 1,
    'Open_circuit': 2,
    'Short': 3,
    'Spur': 4,
    'Spurious_copper': 5
}

# xml annotation parser, converts bounding box coordinates
# to YOLO format (cx, cy, w, h) and normalizes them to [0, 1]

def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_MAP:
            continue
            
        class_id = CLASS_MAP[name]
        bndbox = obj.find('bndbox')
        
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        bboxes.append([class_id, xmin, ymin, xmax, ymax])
        
    return bboxes

# intersection over union to compute the overlap between two bounding boxes
# if the overlap is more than 0.3, keep the bounding box
# otherwise, drop it
def compute_iou(box1, box2):
    # box1: [xmin, ymin, xmax, ymax]
    # box2: [xmin, ymin, xmax, ymax]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    
    # calculate intersection over box1 area to see how much of box1 is inside box2 (the patch)
    return intersection_area / box1_area

# tile image with overlap and save tiles to output directory
def tile_image(image_path, xml_path, out_img_dir, out_label_dir, tile_size=640, overlap=0.15, background_ratio=0.1):
    img = cv2.imread(str(image_path))
    if img is None:
        return
        
    h, w, _ = img.shape
    bboxes = parse_voc_xml(xml_path)
    
    stride = int(tile_size * (1 - overlap))
    
    img_name = Path(image_path).stem
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x1 = x
            y1 = y
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            
            # If the patch is smaller than tile_size, pad or shift it
            if x2 - x1 < tile_size:
                x1 = max(0, w - tile_size)
                x2 = w
            if y2 - y1 < tile_size:
                y1 = max(0, h - tile_size)
                y2 = h
                
            patch = img[y1:y2, x1:x2]
            patch_bboxes = []
            
            patch_box = [x1, y1, x2, y2]
            
            for bbox in bboxes:
                class_id, bx1, by1, bx2, by2 = bbox
                
                # Check if bounding box intersects with patch
                box_area = [bx1, by1, bx2, by2]
                overlap_ratio = compute_iou(box_area, patch_box)
                
                # Lowered from 0.3 to 0.1 to ensure defects caught on the extreme edges/corners
                # of the sliding window grid are not dropped.
                if overlap_ratio > 0.1: 
                    # Clip coordinates
                    nx1 = max(bx1, x1) - x1
                    ny1 = max(by1, y1) - y1
                    nx2 = min(bx2, x2) - x1
                    ny2 = min(by2, y2) - y1
                    
                    # Convert to YOLO format (normalized cx, cy, w, h)
                    cx = (nx1 + nx2) / 2.0 / tile_size
                    cy = (ny1 + ny2) / 2.0 / tile_size
                    bw = (nx2 - nx1) / tile_size
                    bh = (ny2 - ny1) / tile_size
                    
                    # Ensure within [0, 1]
                    cx, cy = max(0, min(cx, 1)), max(0, min(cy, 1))
                    bw, bh = max(0, min(bw, 1)), max(0, min(bh, 1))
                    
                    if bw > 0 and bh > 0:
                        patch_bboxes.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        
            # Save logic: save all patches with defects, save only `background_ratio` of empty patches
            is_empty = len(patch_bboxes) == 0
            if is_empty and random.random() > background_ratio:
                continue
                
            patch_filename = f"{img_name}_{x1}_{y1}.jpg"
            label_filename = f"{img_name}_{x1}_{y1}.txt"
            
            cv2.imwrite(os.path.join(out_img_dir, patch_filename), patch)
            
            with open(os.path.join(out_label_dir, label_filename), 'w') as f:
                f.write("\n".join(patch_bboxes))
                
# process dataset, split into train and val
# 80-20 split
def process_dataset(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    raw_dir = Path(config['raw_data_dir'])
    processed_dir = Path(config['processed_data_dir'])
    
    tile_size = config.get('tile_size', 640)
    overlap = config.get('overlap', 0.15)
    
    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(processed_dir / 'images' / split, exist_ok=True)
        os.makedirs(processed_dir / 'labels' / split, exist_ok=True)
        
    # Get all image paths
    image_paths = list(raw_dir.glob('images/*/*.jpg'))
    random.shuffle(image_paths)
    
    # 80-20 split
    split_idx = int(len(image_paths) * 0.8)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    for split_name, paths in [('train', train_paths), ('val', val_paths)]:
        out_img_dir = processed_dir / 'images' / split_name
        out_label_dir = processed_dir / 'labels' / split_name
        
        print(f"Processing {split_name} split...")
        for img_path in tqdm(paths):
            class_dir = img_path.parent.name
            xml_name = img_path.with_suffix('.xml').name
            xml_path = raw_dir / 'Annotations' / class_dir / xml_name
            
            # Create subdirectories for each defect class inside the train/val split
            class_img_dir = out_img_dir / class_dir
            class_label_dir = out_label_dir / class_dir
            
            os.makedirs(class_img_dir, exist_ok=True)
            os.makedirs(class_label_dir, exist_ok=True)
            
            if xml_path.exists():
                tile_image(img_path, xml_path, class_img_dir, class_label_dir, tile_size, overlap)

if __name__ == '__main__':
    # Test script locally
    process_dataset('configs/dataset.yaml')
