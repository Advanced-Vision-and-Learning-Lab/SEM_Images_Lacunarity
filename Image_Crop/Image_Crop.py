import os
from PIL import Image

def process_image(image_path, output_path):
    with Image.open(image_path) as img:
        # Convert to grayscale and crop height to 1024
        img = img.crop((0, 0, img.width, 1024))
        img.save(output_path, "TIFF")

def process_folders(input_main_folder, output_main_folder):
    # Create output main folder
    os.makedirs(output_main_folder, exist_ok=True)
    
    # Process each subfolder
    for subfolder in os.listdir(input_main_folder):
        input_subfolder = os.path.join(input_main_folder, subfolder)
        if not os.path.isdir(input_subfolder):
            continue
            
        # Create corresponding output subfolder
        output_subfolder = os.path.join(output_main_folder, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)
        
        # Process images in subfolder
        for image_file in os.listdir(input_subfolder):
            if image_file.lower().endswith(('png', 'jpg', 'jpeg', 'tif')):
                input_path = os.path.join(input_subfolder, image_file)
                output_path = os.path.join(output_subfolder, f"{os.path.splitext(image_file)[0]}.tif")
                process_image(input_path, output_path)

# Main execution
input_folder = 'Datasets/Lung Cells SEM Images_group1_DC_Processed'
output_folder = 'Datasets/Lung Cells SEM Images_group1_DC_Processed_v2'
process_folders(input_folder, output_folder)
print("Processing complete.")