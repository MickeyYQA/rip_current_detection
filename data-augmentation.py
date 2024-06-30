import cv2
import numpy as np
import os
import random

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def augment_image(image):
    rows, cols = image.shape[:2]
    
    # Random Rotation by multiples of 90 degrees
    angle = random.choice([0, 90, 180, 270])  # Random angle: 0, 90, 180, 270 degrees
    if angle != 0:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    
    # Random Crop and Resize
    max_crop = 0.05  # Up to 10% of the image can be cropped
    start_row = int(random.uniform(0, max_crop) * rows)
    start_col = int(random.uniform(0, max_crop) * cols)
    end_row = int(rows - random.uniform(0, max_crop) * rows)
    end_col = int(cols - random.uniform(0, max_crop) * cols)
    cropped_image = image[start_row:end_row, start_col:end_col]
    image = cv2.resize(cropped_image, (cols, rows))
    
    # Random Horizontal Flip
    if random.choice([True, False]):
        image = cv2.flip(image, 1)
    
    return image

def process_and_save_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images = load_images_from_folder(input_folder)
    
    for idx, img in enumerate(images):
        augmented_img = augment_image(img)
        #augmented_img = img
        gray_img = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_img, (20, 20))
        
        output_path = os.path.join(output_folder, f"processed_{idx}.png")
        cv2.imwrite(output_path, resized_img)
        print(f"Processed and saved image: {output_path}")

input_folder_path = 'splitted_data/without-rip/train'
output_folder_path = 'splitted_gray/without-rip/train/augmentation-8'
process_and_save_images(input_folder_path, output_folder_path)
