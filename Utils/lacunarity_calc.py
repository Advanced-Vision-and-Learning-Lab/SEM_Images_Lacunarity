import os
import csv
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms as T

from Base_Lacunarity import Base_Lacunarity
from DBC_Lacunarity import DBC_Lacunarity
from Multi_Scale_Lacunarity import MS_Lacunarity

# Path to the output main folder where the divided images are saved
output_main_folder_path = 'Datasets/Lung_Cells_DC_Split_overlap/'

# Path to the CSV file where the results will be saved
csv_file_path = 'C:/Users/aksha/Peeples_Lab/SEM_Images_Lacunarity/Datasets/DC_changed_lacunarity_results_all.csv'

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.379, std=0.224)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.379, std=0.224)
    ]),
}

# List to store the results
results = []

# Define classes
class_mapping = {
    "Lung Cells Exposed to Crystalline Silica (CS)": "Crystalline Silica (CS)",
    "Lung Cells Exposed to Isocyanate (IPDI)": "Isocyanate (IPDI)",
    "Lung Cells Exposed to Nickel Oxide (NiO)": "Nickel Oxide (NiO)",
    "Lung Cells Exposed to Silver Nanoparticles (Ag-NP)": "Silver Nanoparticles (Ag-NP)",
    "Lung Cells Untreated": "Untreated"
}

# Loop through 'train' and 'val' folders
for split in ['train', 'val']:
    split_path = os.path.join(output_main_folder_path, split)
    if os.path.isdir(split_path):
        print(f"Processing {split} folder...")
        
        # Loop through each class subfolder
        for class_folder in os.listdir(split_path):
            class_folder_path = os.path.join(split_path, class_folder)
            if os.path.isdir(class_folder_path):
                print(f"Processing class folder: {class_folder}")
                
                # Process each image in the class folder
                for image_file in os.listdir(class_folder_path):
                    if image_file.lower().endswith(('png', 'jpg', 'jpeg', 'tif')):
                        image_path = os.path.join(class_folder_path, image_file)
                        print(f"Calculating lacunarity for image: {image_file}")

                        # Determine the class based on the image path
                        if "Crystalline Silica (CS)" in image_path:
                            class_name = "Crystalline Silica (CS)"
                        elif "Isocyanate (IPDI)" in image_path:
                            class_name = "Isocyanate (IPDI)"
                        elif "Nickel Oxide (NiO)" in image_path:
                            class_name = "Nickel Oxide (NiO)"
                        elif "Silver Nanoparticles (Ag-NP)" in image_path:
                            class_name = "Silver Nanoparticles (Ag-NP)"
                        else:
                            class_name = "Untreated"
                        
                        # Open the image
                        img = Image.open(image_path)
                        image = np.array(img)
                        image = (image / image.max()) * 255
                        image = image.astype(np.uint8)
                        to_pil = T.ToPILImage()
                        image = to_pil(image)
                        
                        # Calculate lacunarity
                        transform = data_transforms[split]
                        tensor = transform(image)
                        
                        base_lacunarity = Base_Lacunarity(kernel=None)
                        dbc_lacunarity = DBC_Lacunarity(window_size=224)
                        multi_lacunarity = MS_Lacunarity(num_levels=3)
                        dbc_lacunarity_value = dbc_lacunarity(tensor)
                        base_lacunarity_value = base_lacunarity(tensor)
                        multi_lacunarity_value = multi_lacunarity(tensor.unsqueeze(0))
                        
                        # Append the result to the list
                        results.append([class_name, base_lacunarity_value.item(), dbc_lacunarity_value.item(), multi_lacunarity_value.item()])

# Write the results to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Class", "Base_Lacunarity", "DBC_Lacunarity", "Multi_Lacunarity"])
    writer.writerows(results)

print("Lacunarity calculation complete and results saved to CSV file.")
