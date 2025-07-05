import cv2
import numpy as np
import os
from glob import glob

input_dir = 'car_images'
output_dir = 'boxed_images'  # Contains only cropped license plates
os.makedirs(output_dir, exist_ok=True)

image_paths = glob(os.path.join(input_dir, '*.jpg'))

for img_path in image_paths:
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_name}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhancement (simulating imadjust)
    enh = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    enh = cv2.convertScaleAbs(enh, alpha=1.5, beta=0)

    # Thresholding
    black_region = (enh < 80).astype(np.uint8) * 255

    # Noise removal (bottom half only)
    height = black_region.shape[0]
    noise_remov = np.zeros_like(black_region)
    noise_remov[height // 2:, :] = black_region[height // 2:, :]

    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(noise_remov, connectivity=8)
    min_area = 300
    noise_cleaned = np.zeros_like(noise_remov)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            noise_cleaned[labels == i] = 255

    # Detect candidate bounding boxes
    contours, _ = cv2.findContours(noise_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plate_found = False

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h
        roi = noise_cleaned[y:y+h, x:x+w]
        pixel_ratio = np.sum(roi) / 255 / (w * h)
        tpx = np.sum(roi) / 255

        if abs(aspect - 2.4) < 0.2 and abs(pixel_ratio - 0.7) < 0.6 and abs(tpx - 120000) < 55000:
            # Crop the license plate region from original image
            cropped_plate = img[y:y+h, x:x+w]
            plate_found = True
            break  # Only one plate

    if plate_found:
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, cropped_plate)
        print(f"[✔] Cropped and saved plate: {output_path}")
    else:
        print(f"[✘] No plate detected in: {img_name}")
