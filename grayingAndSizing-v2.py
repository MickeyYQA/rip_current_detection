import cv2
import os

def ensure_folder_exists(folder_path):
    """Check if the folder exists, and create it if it doesn't."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

def convert_and_resize_images(input_dir, output_dir, target_size):
    """
    Convert color images to grayscale and resize them.

    Parameters:
    input_dir (str): The directory containing the input images.
    output_dir (str): The directory where the processed images will be saved.
    target_size (tuple): The target size (width, height) for the images.
    """
    ensure_folder_exists(output_dir)
    
    # Get a list of all files in the input directory
    image_files = [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))]

    # Process each image
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        # Read the color image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Cannot read image: {input_path}")
            continue
        
        # Convert the color image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize the grayscale image
        resized_image = cv2.resize(gray_image, target_size)
        
        # Save the processed image
        cv2.imwrite(output_path, resized_image)
        print(f"Processed and saved image: {output_path}")

    print("All images have been processed successfully.")

# Define the input and output directories and target size
input_directory = "training_data/without_rips"
output_directory = "training_data/without_rips_gray_10"
target_size = (10, 10)

# Call the function to process images
convert_and_resize_images(input_directory, output_directory, target_size)