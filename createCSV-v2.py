import os
import csv
from PIL import Image
import numpy as np

def get_image_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.png') and f.startswith('rip-') and f[4:-4].isdigit()]

def process_images(input_folder, output_csv):
    # Get all the image files
    image_files = get_image_files(input_folder)
    
    # Sort the files by their numerical names
    image_files.sort(key=lambda x: int(x[4:-4]))
    
    # Prepare the output data
    output_data = []

    # Read each image and convert it to grayscale values
    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)
        with Image.open(img_path) as img:
            img = img.convert('L')  # Convert image to grayscale
            pixels = np.array(img).flatten()
            row = [1] + pixels.tolist()
            output_data.append(row)
    
    # Write the output to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([''] + [str(i) for i in range(len(output_data[0]) - 1)])  # Write the header row
        writer.writerows(output_data)

# Define the input folder and output CSV file
input_folder = 'training_data/with_rips_gray_10pix'
output_csv = 'output.csv'

# Process the images and write to CSV
process_images(input_folder, output_csv)
