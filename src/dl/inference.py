import cv2
import numpy as np
import torch
from ultralytics import YOLO
import torchvision

class BaseInferencePipeline:
    def run_inference(self, img):
        raise NotImplementedError("Subclasses must implement this method")

class SingleStageYOLOPipeline(BaseInferencePipeline):
    def __init__(self, model_path, tile_size=640, overlap=0.15, conf_thresh=0.25, iou_thresh=0.45):
        self.model = YOLO(model_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
    def tile_image(self, img):
        h, w, _ = img.shape
        stride = int(self.tile_size * (1 - self.overlap))
        
        tiles = []
        coords = []
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x1 = x
                y1 = y
                x2 = min(x + self.tile_size, w)
                y2 = min(y + self.tile_size, h)
                
                if x2 - x1 < self.tile_size:
                    x1 = max(0, w - self.tile_size)
                    x2 = w
                if y2 - y1 < self.tile_size:
                    y1 = max(0, h - self.tile_size)
                    y2 = h
                    
                patch = img[y1:y2, x1:x2]
                tiles.append(patch)
                coords.append((x1, y1, x2, y2))
                
        return tiles, coords
        
    def run_inference(self, img):
        tiles, coords = self.tile_image(img)
        
        all_boxes = []
        all_scores = []
        all_classes = []
        
        # Run inference on each tile
        for idx, tile in enumerate(tiles):
            x1_offset, y1_offset, _, _ = coords[idx]
            
            results = self.model(tile, conf=self.conf_thresh, verbose=False)[0]
            
            # Map bounding boxes back to original image coordinates
            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                
                for i in range(len(boxes)):
                    bx1, by1, bx2, by2 = boxes[i]
                    
                    # Remap to full image
                    orig_bx1 = bx1 + x1_offset
                    orig_by1 = by1 + y1_offset
                    orig_bx2 = bx2 + x1_offset
                    orig_by2 = by2 + y1_offset
                    
                    all_boxes.append([orig_bx1, orig_by1, orig_bx2, orig_by2])
                    all_scores.append(scores[i])
                    all_classes.append(classes[i])
                    
        if len(all_boxes) == 0:
            return [], [], []
            
        # Apply Global NMS to remove duplicate boxes along seams
        all_boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        all_scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
        all_classes_tensor = torch.tensor(all_classes, dtype=torch.float32)
        
        # torchvision NMS is class-agnostic by default, but we want class-aware NMS
        # offset boxes by class id to perform class-aware NMS
        max_coord = all_boxes_tensor.max()
        offsets = all_classes_tensor * (max_coord + 1)
        boxes_for_nms = all_boxes_tensor + offsets[:, None]
        
        keep_indices = torchvision.ops.nms(boxes_for_nms, all_scores_tensor, self.iou_thresh)
        
        final_boxes = all_boxes_tensor[keep_indices].numpy()
        final_scores = all_scores_tensor[keep_indices].numpy()
        final_classes = all_classes_tensor[keep_indices].numpy()
        
        return final_boxes, final_scores, final_classes

class TwoStageYOLOPipeline(BaseInferencePipeline):
    def __init__(self, yolo_model_path, classifier_model_path=None, tile_size=640, overlap=0.15, conf_thresh=0.25, iou_thresh=0.45):
        # We will load YOLO here to do the cropping
        self.yolo_model = YOLO(yolo_model_path)
        self.classifier_model_path = classifier_model_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def run_inference(self, img):
        # TODO: Implement 1. YOLO Region Proposal -> 2. Crop -> 3. Pass to Custom CNN
        raise NotImplementedError("Two-stage CNN classification coming soon")

def get_inference_pipeline(method="single_stage", **kwargs):
    if method == "single_stage":
        return SingleStageYOLOPipeline(**kwargs)
    elif method == "two_stage":
        return TwoStageYOLOPipeline(**kwargs)
    else:
        raise ValueError(f"Unknown inference pipeline method: {method}")
