import cv2
import sys
import matplotlib.pyplot as plt
import os
from src.classical.template_matching import detect_defects

def main():
    test_path = sys.argv[1]
    temp_path = sys.argv[2]
    out_path = "classical_output.jpg"
    
    test_img = cv2.imread(test_path)
    template_img = cv2.imread(temp_path)
    
    if test_img is None or template_img is None:
        print("Error reading images")
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
        
    plt.savefig(out_path)
    print(f"Saved output to {out_path}")

if __name__ == '__main__':
    main()
