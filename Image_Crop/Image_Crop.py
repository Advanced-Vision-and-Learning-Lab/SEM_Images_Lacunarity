import os
from PIL import Image
import pdb


# Function to divide an image into smaller sections without exceeding original size
def divide_image(image_path, output_folder, image_name, tile_size=(768, 512)):
    img = Image.open(image_path)
    img = img.crop((0, 0, img.width, 1024))
    img = img.convert("L")  # Convert to grayscale
    img_width, img_height = img.size
    save_path = os.path.join(output_folder, f"{image_name}.png")
    img.save(save_path)
    print(f"Saved image: {save_path}")

# Path to the main folder containing the subfolders with images
main_folder_path = 'Datasets/Lung Cells SEM Images_group1_DC_NEW/'
# Path to the new main output folder
output_main_folder_path = 'Datasets/Lung Cells Untreated'

# Create the main output folder if it doesn't exist
os.makedirs(output_main_folder_path, exist_ok=True)

# Loop through each subfolder and process the images
for subfolder in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        output_subfolder_path = os.path.join(output_main_folder_path, subfolder)
        os.makedirs(output_subfolder_path, exist_ok=True)
        
        print(f"Processing subfolder: {subfolder}")
        
        for image_file in os.listdir(subfolder_path):
            if image_file.lower().endswith(('png', 'jpg', 'jpeg', 'tif')):
                image_path = os.path.join(subfolder_path, image_file)
                image_name, _ = os.path.splitext(image_file)
                print(f"Dividing image: {image_file}")
                divide_image(image_path, output_subfolder_path, image_name)

print("Processing complete.")
