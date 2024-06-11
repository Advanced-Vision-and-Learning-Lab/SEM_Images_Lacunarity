import os
from PIL import Image

# Function to divide an image into smaller sections with 50% overlap
def divide_image(image_path, output_folder, image_name, tile_size=(768, 512), overlap=0.5):
    img = Image.open(image_path)
    img = img.crop((0, 0, img.width, 1024))  # Ensure the image height is 1024
    img_width, img_height = img.size

    count = 0
    step_x = int(tile_size[0] * (1 - overlap))  # 50% of tile width (384)
    step_y = int(tile_size[1] * (1 - overlap))  # 50% of tile height (256)

    for y in range(0, img_height - tile_size[1] + 1, step_y):
        for x in range(0, img_width - tile_size[0] + 1, step_x):
            box = (x, y, x + tile_size[0], y + tile_size[1])
            cropped_img = img.crop(box)
            save_path = os.path.join(output_folder, f"{image_name}_{count}.png")
            cropped_img.save(save_path)
            print(f"Saved image: {save_path}")
            count += 1

    # Handle the right and bottom edges
    for y in range(0, img_height - tile_size[1] + 1, step_y):
        box = (img_width - tile_size[0], y, img_width, y + tile_size[1])
        cropped_img = img.crop(box)
        save_path = os.path.join(output_folder, f"{image_name}_{count}.png")
        cropped_img.save(save_path)
        print(f"Saved image: {save_path}")
        count += 1

    for x in range(0, img_width - tile_size[0] + 1, step_x):
        box = (x, img_height - tile_size[1], x + tile_size[0], img_height)
        cropped_img = img.crop(box)
        save_path = os.path.join(output_folder, f"{image_name}_{count}.png")
        cropped_img.save(save_path)
        print(f"Saved image: {save_path}")
        count += 1

    # Handle the bottom-right corner
    box = (img_width - tile_size[0], img_height - tile_size[1], img_width, img_height)
    cropped_img = img.crop(box)
    save_path = os.path.join(output_folder, f"{image_name}_{count}.png")
    cropped_img.save(save_path)
    print(f"Saved image: {save_path}")
    count += 1

# Path to the main folder containing the subfolders with images
main_folder_path = 'Datasets/Lung Cells SEM Images_group1_DC'
# Path to the new main output folder
output_main_folder_path = 'Datasets/Lung Cells SEM Images_group1_DC_overlap_NEW'

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
