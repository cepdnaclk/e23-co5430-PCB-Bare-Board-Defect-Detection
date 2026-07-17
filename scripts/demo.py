import cv2
import argparse
from pathlib import Path
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
        
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"YOLO Defect Detection ({len(boxes)} defects found)")
        plt.axis('off')
        plt.show()
        
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
        
        out_img = aligned.copy()
        for box in bboxes:
            x, y, w, h = box
            cv2.rectangle(out_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
        fig, axs = plt.subplots(1, 3, figsize=(20, 8))
        axs[0].imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Aligned Test Image")
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Defect Mask")
        axs[2].imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
        axs[2].set_title(f"Detected Defects ({len(bboxes)})")
        
        for ax in axs:
            ax.axis('off')
            
        plt.show()

if __name__ == "__main__":
    main()
