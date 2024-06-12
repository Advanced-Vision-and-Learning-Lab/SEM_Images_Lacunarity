import os
import csv
from PIL import Image
from Base_Lacunarity import Base_Lacunarity
from DBC_Lacunarity import DBC_Lacunarity
import pdb
import numpy as np
import torchvision.transforms as transforms


# Path to the output main folder where the divided images are saved

output_main_folder_path = 'C:/Users/aksha/Peeples_Lab/SEM_Images_Lacunarity/Datasets/Lung Cells SEM Images_group1_DC_NEW'

# Path to the CSV file where the results will be saved
csv_file_path = 'C:/Users/aksha/Peeples_Lab/SEM_Images_Lacunarity/Datasets/DC_lacunarity_results.csv'

# List to store the results
results = []

# Loop through each subfolder and process the images
for subfolder in os.listdir(output_main_folder_path):
    subfolder_path = os.path.join(output_main_folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        print(f"Processing subfolder: {subfolder}")
        
        for image_file in os.listdir(subfolder_path):
            if image_file.lower().endswith(('png', 'jpg', 'jpeg', 'tif')):
                image_path = os.path.join(subfolder_path, image_file)
                print(f"Calculating lacunarity for image: {image_file}")
                
                # Open the image
                img = Image.open(image_path)
                
                # Calculate lacunarity
                transform = transforms.ToTensor()
                tensor = transform(img)
                base_lacunarity = Base_Lacunarity(kernel=None)

                dbc_lacunarity = DBC_Lacunarity(window_size = tuple(tensor.squeeze(0).size()))
                dbc_lacunarity_value = dbc_lacunarity(tensor)
                base_lacunarity_value = base_lacunarity(tensor)
                
                # Append the result to the list
                results.append([image_path, base_lacunarity_value, dbc_lacunarity_value])

# Write the results to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Path", "Base_Lacunarity", "DBC_Lacunarity"])
    writer.writerows(results)

print("Lacunarity calculation complete and results saved to CSV file.")


