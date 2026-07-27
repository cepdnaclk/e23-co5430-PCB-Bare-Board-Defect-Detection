import cv2
import numpy as np
import yaml
from pathlib import Path
import os
import glob

def align_images(im1, im2):
    """
    Align im1 (test image) to im2 (reference template) using ORB and Homography.
    Attempts 4 rotations (0, 90, 180, 270) to handle augmented data.
    """
    MAX_FEATURES = 5000
    GOOD_MATCH_PERCENT = 0.15

    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    im2_gray = clahe.apply(im2_gray)
    
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)
    
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    
    best_matches = []
    best_h = None
    best_im1_reg = None
    
    # Try all 4 rotations
    for k in range(4):
        rotated_im1 = im1.copy()
        if k == 1:
            rotated_im1 = cv2.rotate(im1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif k == 2:
            rotated_im1 = cv2.rotate(im1, cv2.ROTATE_180)
        elif k == 3:
            rotated_im1 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)
            
        im1_gray = cv2.cvtColor(rotated_im1, cv2.COLOR_BGR2GRAY)
        im1_gray = clahe.apply(im1_gray)
        
        keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
        
        if descriptors1 is None or descriptors2 is None:
            continue
            
        matches = matcher.match(descriptors1, descriptors2, None)
        matches = list(matches)
        matches.sort(key=lambda x: x.distance, reverse=False)
        
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]
        
        if len(matches) > len(best_matches):
            best_matches = matches
            
            # Find homography for this rotation
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)

            for i, match in enumerate(matches):
                points1[i, :] = keypoints1[match.queryIdx].pt
                points2[i, :] = keypoints2[match.trainIdx].pt

            h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
            
            if h is not None:
                height, width, channels = im2.shape
                # Warp the rotated image
                best_im1_reg = cv2.warpPerspective(rotated_im1, h, (width, height))
                best_h = h
                
    if best_im1_reg is None:
        # Fallback
        best_im1_reg = im1
        best_h = np.eye(3)
        
    return best_im1_reg, best_h

def classify_defect(x, y, w, h, test_gray, template_gray):
    """
    Heuristic rule-based classification based on Additive vs Subtractive intensity.
    """
    test_roi = test_gray[y:y+h, x:x+w]
    template_roi = template_gray[y:y+h, x:x+w]
    
    test_mean = np.mean(test_roi)
    template_mean = np.mean(template_roi)
    
    if test_mean > template_mean:
        # Additive Defect (Excess Copper). Mapping to Spurious_copper (5)
        return 5
    else:
        # Subtractive Defect (Missing Copper). Mapping to Open_circuit (2)
        return 2

def detect_defects(test_img, template_img):
    """
    Detect defects by absolute subtraction and morphological thresholding.
    Returns the binary defect mask and list of bounding boxes [x, y, w, h, class_id].
    """
    aligned_test, _ = align_images(test_img, template_img)
    
    test_gray = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    test_gray = clahe.apply(test_gray)
    template_gray = clahe.apply(template_gray)
    
    # test_gray = cv2.GaussianBlur(test_gray, (5, 5), 0)
    # template_gray = cv2.GaussianBlur(template_gray, (5, 5), 0)

    diff = cv2.absdiff(test_gray, template_gray)
    
    # Otsu's Binarization (Adaptive Thresholding)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5,5), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes_with_classes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 110:  # Minimum area to be considered a defect
            x, y, w, h = cv2.boundingRect(cnt)
            class_id = classify_defect(x, y, w, h, test_gray, template_gray)
            bboxes_with_classes.append([x, y, w, h, class_id])
            
    return aligned_test, closed, bboxes_with_classes

def test_classical_pipeline(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = Path(config.get('classical_data_dir', ''))
    
    image_paths = list(data_dir.glob('valid/images/*.jpg'))
    if not image_paths:
        image_paths = list(data_dir.glob('images/*/*.jpg'))
        
    out_dir = Path("runs/detect/classical_predictions/labels")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running batch classical inference on {len(image_paths)} images...")
    
    for img_path in image_paths:
        test_img = cv2.imread(str(img_path))
        prefix = img_path.name.split('_')[0]
        template_path = data_dir / 'PCB_USED' / f"{prefix}.JPG"
        
        if not template_path.exists() or test_img is None:
            continue
            
        template_img = cv2.imread(str(template_path))
        if template_img is None:
            continue
            
        _, _, bboxes = detect_defects(test_img, template_img)
        
        height, width = test_img.shape[:2]
        txt_path = out_dir / img_path.with_suffix('.txt').name
        
        with open(txt_path, 'w') as f:
            for box in bboxes:
                x, y, w, h, cls_id = box
                x_center = (x + w/2) / width
                y_center = (y + h/2) / height
                nw = w / width
                nh = h / height
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}\n")
    print(f"Batch inference complete. Saved to {out_dir}")

if __name__ == "__main__":
    test_classical_pipeline("configs/dataset.yaml")
