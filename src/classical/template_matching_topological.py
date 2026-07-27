import cv2
import numpy as np

def align_images(im1, im2):
    """
    Align im1 (test image) to im2 (reference template) using ORB and Homography.
    Attempts 4 rotations (0, 90, 180, 270).
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
            
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)

            for i, match in enumerate(matches):
                points1[i, :] = keypoints1[match.queryIdx].pt
                points2[i, :] = keypoints2[match.trainIdx].pt

            h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
            
            if h is not None:
                height, width, channels = im2.shape
                best_im1_reg = cv2.warpPerspective(rotated_im1, h, (width, height))
                best_h = h
                
    if best_im1_reg is None:
        best_im1_reg = im1
        best_h = np.eye(3)
        
    return best_im1_reg, best_h


def classify_defect_topological(x, y, w, h, cnt, test_gray, template_gray):
    """
    Advanced topological classification using intersection counting.
    """
    margin = 25
    h_img, w_img = test_gray.shape
    y1, y2 = max(0, y - margin), min(h_img, y + h + margin)
    x1, x2 = max(0, x - margin), min(w_img, x + w + margin)
    
    test_roi = test_gray[y1:y2, x1:x2]
    template_roi = template_gray[y1:y2, x1:x2]
    
    # Get Copper Masks for the ROI using Otsu
    _, test_copper = cv2.threshold(test_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, template_copper = cv2.threshold(template_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Draw the specific defect contour mask
    defect_mask = np.zeros_like(test_roi)
    shifted_cnt = cnt - [x1, y1]
    cv2.drawContours(defect_mask, [shifted_cnt], -1, 255, thickness=cv2.FILLED)
    
    # 1. Missing Hole Check (Template Copper Ring)
    # The user correctly noted that a missing hole has a metal ring around it in the template.
    # We can detect this topological ring using contour hierarchy (RETR_TREE).
    t_cnts, hierarchy = cv2.findContours(template_copper, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            # Defect centroid (in ROI coordinates)
            cx = int(M["m10"] / M["m00"]) - x1
            cy = int(M["m01"] / M["m00"]) - y1
            
            for i in range(len(t_cnts)):
                child_idx = hierarchy[0][i][2]
                if child_idx != -1: # This contour has a hole inside it
                    child_cnt = t_cnts[child_idx]
                    # If the defect centroid is inside this hole, it's a Missing Hole!
                    if cv2.pointPolygonTest(child_cnt, (float(cx), float(cy)), False) >= 0:
                        return 0 # Missing Hole
    
    # Dilate defect to overlap adjacent copper traces
    # kernel = np.ones((5,5), np.uint8)
    kernel = np.ones((11,11), np.uint8)
    dilated_defect = cv2.dilate(defect_mask, kernel, iterations=1)
    
    # Intensity check
    test_mean = np.mean(test_gray[y:y+h, x:x+w])
    template_mean = np.mean(template_gray[y:y+h, x:x+w])
    
    if test_mean > template_mean: # Additive Defect
        intersection = cv2.bitwise_and(dilated_defect, template_copper)
        num_labels, _ = cv2.connectedComponents(intersection)
        count = num_labels - 1 # Exclude background
        
        if count >= 2: return 3 # Short (bridges 2+ traces)
        elif count == 1: return 4 # Spur (attached to 1 trace)
        else: return 5 # Spurious Copper (isolated)
        
    else: # Subtractive Defect
        # 2. Open Circuit vs Mouse Bite (Stump counting)
        intersection = cv2.bitwise_and(dilated_defect, test_copper)
        num_labels, _ = cv2.connectedComponents(intersection)
        count = num_labels - 1
        
        if count >= 2: return 2 # Open Circuit (Trace severed into 2+ pieces)
        else: return 1 # Mouse Bite (Chipped off edge of 1 trace)

def detect_defects_topological(test_img, template_img):
    aligned_test, _ = align_images(test_img, template_img)
    
    test_gray = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    test_gray = clahe.apply(test_gray)
    template_gray = clahe.apply(template_gray)
    
    diff = cv2.absdiff(test_gray, template_gray)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5,5), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes_with_classes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 130:
            x, y, w, h = cv2.boundingRect(cnt)
            class_id = classify_defect_topological(x, y, w, h, cnt, test_gray, template_gray)
            bboxes_with_classes.append([x, y, w, h, class_id])
            
    return aligned_test, closed, bboxes_with_classes
