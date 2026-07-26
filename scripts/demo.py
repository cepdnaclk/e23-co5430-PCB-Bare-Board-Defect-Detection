import sys
import cv2
import argparse
from pathlib import Path

# Add project root to sys.path so it can find 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dl.inference import YOLOInferencePipeline
from src.classical.template_matching import detect_defects
import matplotlib.pyplot as plt

def draw_boxes(img, boxes, scores, classes, class_names=None):
    out_img = img.copy()
    for i in range(len(boxes)):
        bx1, by1, bx2, by2 = map(int, boxes[i])
        score = scores[i]
        cls_id = int(classes[i])
        
        name = class_names[cls_id] if class_names else str(cls_id)
        label = f"{name}: {score:.2f}"
        
        cv2.rectangle(out_img, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
        cv2.putText(out_img, label, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
    return out_img

def main():
    parser = argparse.ArgumentParser(description="MicroInspect Demo")
    parser.add_argument('--test_img', type=str, required=True, help="Path to test image")
    parser.add_argument('--template_img', type=str, help="Path to template image (for classical method)")
    parser.add_argument('--method', type=str, choices=['dl', 'classical'], default='dl', help="Method to use")
    parser.add_argument('--model', type=str, default='runs/train/microinspect_yolo/weights/best.pt', help="Path to YOLO model")
    
    args = parser.parse_args()
    
    test_img = cv2.imread(args.test_img)
    if test_img is None:
        print("Could not read test image.")
        return
        
    if args.method == 'dl':
        print("Running Deep Learning pipeline...")
        class_names = {0: 'Missing_hole', 1: 'Mouse_bite', 2: 'Open_circuit', 3: 'Short', 4: 'Spur', 5: 'Spurious_copper'}
        
        try:
            pipeline = YOLOInferencePipeline(args.model)
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
            
        boxes, scores, classes = pipeline.run_inference(test_img)
        result_img = draw_boxes(test_img, boxes, scores, classes, class_names)
        
        # Create output directory
        test_img_name = Path(args.test_img).stem
        project_root = Path(__file__).resolve().parent.parent.parent
        out_dir = project_root / "outputs" / "dl" / test_img_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save output image
        out_path = out_dir / f"{test_img_name}_result.jpg"
        cv2.imwrite(str(out_path), result_img)
        print(f"Saved YOLO output to {out_path}")
        
    elif args.method == 'classical':
        print("Running Classical CV pipeline...")
        if not args.template_img:
            print("Template image is required for classical method.")
            return
            
        template_img = cv2.imread(args.template_img)
        if template_img is None:
            print("Could not read template image.")
            return
            
        aligned, mask, bboxes = detect_defects(test_img, template_img)
        
        class_names = {0: 'Missing_hole', 1: 'Mouse_bite', 2: 'Open_circuit', 3: 'Short', 4: 'Spur', 5: 'Spurious_copper'}
        out_img = aligned.copy()
        for box in bboxes:
            if len(box) == 5:
                x, y, w, h, class_id = box
            else:
                x, y, w, h = box
                class_id = 0
                
            name = class_names.get(class_id, "Unknown")
            cv2.rectangle(out_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(out_img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        # Create output directory
        test_img_name = Path(args.test_img).stem
        project_root = Path(__file__).resolve().parent.parent.parent
        out_dir = project_root / "outputs" / "classical" / test_img_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual images instead of a combined plot
        aligned_path = out_dir / f"{test_img_name}_aligned.jpg"
        mask_path = out_dir / f"{test_img_name}_mask.jpg"
        result_path = out_dir / f"{test_img_name}_result.jpg"
        
        cv2.imwrite(str(aligned_path), aligned)
        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(result_path), out_img)
        
        print(f"Saved individual images to {out_dir}")

if __name__ == "__main__":
    main()
