import cv2
import numpy as np
import yaml
from pathlib import Path
import os
import glob

def align_images(im1, im2):
    """
    Align im1 (test image) to im2 (reference template) using ORB and Homography.
    Tests 4 rotations (0, 90, 180, 270) to find the best match.
    """
    MAX_FEATURES = 5000
    GOOD_MATCH_PERCENT = 0.15

    # Convert template to grayscale and apply CLAHE
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    im2_gray = clahe.apply(im2_gray)

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    best_matches = []
    best_keypoints1 = None
    best_im1_rotated = None

    # Try 4 rotations
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            im1_rotated = im1.copy()
        elif angle == 90:
            im1_rotated = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            im1_rotated = cv2.rotate(im1, cv2.ROTATE_180)
        elif angle == 270:
            im1_rotated = cv2.rotate(im1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        im1_gray = cv2.cvtColor(im1_rotated, cv2.COLOR_BGR2GRAY)
        im1_gray = clahe.apply(im1_gray)
        
        keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
        if descriptors1 is None or descriptors2 is None:
            continue
            
        matches = matcher.match(descriptors1, descriptors2, None)
        matches = list(matches)
        matches.sort(key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        good_matches = matches[:numGoodMatches]

        if len(good_matches) > len(best_matches):
            best_matches = good_matches
            best_keypoints1 = keypoints1
            best_im1_rotated = im1_rotated

    if not best_matches:
        return im1, np.eye(3) # Fallback

    # Extract location of good matches
    points1 = np.zeros((len(best_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(best_matches), 2), dtype=np.float32)

    for i, match in enumerate(best_matches):
        points1[i, :] = best_keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    if h is None:
        return best_im1_rotated, np.eye(3)

    # Use homography
    height, width, channels = im2.shape
    im1_reg = cv2.warpPerspective(best_im1_rotated, h, (width, height))

    return im1_reg, h

def classify_defect(x, y, w, h, test_gray, template_gray):
    """
    Classify the defect based on intensity differences.
    Additive (Test brighter) -> Spurious_copper (5)
    Subtractive (Template brighter) -> Mouse_bite (1)
    """
    test_crop = test_gray[y:y+h, x:x+w]
    template_crop = template_gray[y:y+h, x:x+w]
    
    test_mean = np.mean(test_crop)
    template_mean = np.mean(template_crop)
    
    if test_mean > template_mean:
        return 5  # Spurious_copper
    else:
        return 1  # Mouse_bite

def detect_defects(test_img, template_img, threshold=30):
    """
    Detect defects by absolute subtraction and morphological thresholding.
    Returns the binary defect mask and list of bounding boxes [x, y, w, h, class_id].
    """
    # Align test image to template
    aligned_test, _ = align_images(test_img, template_img)
    
    # Convert to grayscale
    test_gray = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to mitigate lighting variations before absolute difference
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    test_gray_clahe = clahe.apply(test_gray)
    template_gray_clahe = clahe.apply(template_gray)
    
    # Pre-subtraction Gaussian Blur to mitigate slight misalignments
    test_blur = cv2.GaussianBlur(test_gray_clahe, (5, 5), 0)
    template_blur = cv2.GaussianBlur(template_gray_clahe, (5, 5), 0)
    
    # Absolute difference
    diff = cv2.absdiff(test_blur, template_blur)
    
    # Adaptive Thresholding using Otsu's Method
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to remove noise
    kernel = np.ones((5,5), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  # Minimum area to be considered a defect
            x, y, w, h = cv2.boundingRect(cnt)
            class_id = classify_defect(x, y, w, h, test_gray, template_gray)
            bboxes.append([x, y, w, h, class_id])
            
    return aligned_test, closed, bboxes

def test_classical_pipeline(config_path):
    """
    Run the classical pipeline over the dataset and save YOLO-format .txt predictions.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config.get('classical_data_dir', config.get('raw_data_dir', ''))
    raw_dir = Path(data_dir)
    
    out_dir = Path('runs/detect/classical_predictions')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(raw_dir.glob('valid/images/*.jpg'))
    if not image_paths:
        image_paths = list(raw_dir.glob('images/*/*.jpg'))
        
    print(f"Running classical batch inference on {len(image_paths)} images...")
    
    for img_path in image_paths:
        test_img = cv2.imread(str(img_path))
        prefix = img_path.name.split('_')[0]
        template_path = raw_dir / 'PCB_USED' / f"{prefix}.JPG"
        template_img = cv2.imread(str(template_path))
        
        if test_img is None or template_img is None:
            continue
            
        _, _, bboxes = detect_defects(test_img, template_img)
        
        txt_path = out_dir / img_path.with_suffix('.txt').name
        height, width = test_img.shape[:2]
        
        with open(txt_path, 'w') as f:
            for box in bboxes:
                if len(box) == 5:
                    x, y, w, h, class_id = box
                else:
                    x, y, w, h = box
                    class_id = 0
                    
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                norm_w = w / width
                norm_h = h / height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    print(f"Predictions saved to {out_dir}")

if __name__ == "__main__":
    test_classical_pipeline('configs/dataset.yaml')
