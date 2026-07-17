import cv2
import numpy as np
import yaml
from pathlib import Path
import os
import glob

def align_images(im1, im2):
    """
    Align im1 (test image) to im2 (reference template) using ORB and Homography.
    """
    MAX_FEATURES = 5000
    GOOD_MATCH_PERCENT = 0.15

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1_reg = cv2.warpPerspective(im1, h, (width, height))

    return im1_reg, h

def detect_defects(test_img, template_img, threshold=30):
    """
    Detect defects by absolute subtraction and morphological thresholding.
    Returns the binary defect mask and list of bounding boxes [x, y, w, h].
    """
    # Align test image to template
    aligned_test, _ = align_images(test_img, template_img)
    
    # Convert to grayscale
    test_gray = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    
    # Absolute difference
    diff = cv2.absdiff(test_gray, template_gray)
    
    # Thresholding
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
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
            bboxes.append([x, y, w, h])
            
    return aligned_test, closed, bboxes

def test_classical_pipeline(config_path):
    # This is a placeholder test function
    pass

if __name__ == "__main__":
    pass
