import os
import csv
from PIL import Image
from Base_Lacunarity import Base_Lacunarity
from DBC_Lacunarity import DBC_Lacunarity
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms as T


# Path to the output main folder where the divided images are saved
output_main_folder_path = 'Datasets/Lung_Cells_ME_Split_overlap'

# Path to the CSV file where the results will be saved
csv_file_path = 'C:/Users/aksha/Peeples_Lab/SEM_Images_Lacunarity/Datasets/ME_lacunarity_results.csv'

data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224,scale=(.8,1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.379, std=0.224)
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.379, std=0.224)
            ]),
        }
# List to store the results
results = []

# Loop through each subfolder (TRAIN and TEST) and process the images
for subfolder_type in os.listdir(output_main_folder_path):
    subfolder_type_path = os.path.join(output_main_folder_path, subfolder_type)
    if os.path.isdir(subfolder_type_path):
        print(f"Processing {subfolder_type} subfolder...")
        
        # Loop through each class subfolder
        for class_folder in os.listdir(subfolder_type_path):
            class_folder_path = os.path.join(subfolder_type_path, class_folder)
            if os.path.isdir(class_folder_path):
                print(f"Processing class folder: {class_folder}")
                
                # Process each image in the class folder
                for image_file in os.listdir(class_folder_path):
                    if image_file.lower().endswith(('png', 'jpg', 'jpeg', 'tif')):
                        image_path = os.path.join(class_folder_path, image_file)
                        print(f"Calculating lacunarity for image: {image_file}")
                        
                        #Open the image
                        img = Image.open(image_path)
                        image = np.array(img)
                        image = (image/image.max()) * 255
                        image = image.astype(np.uint8)
                        to_pil = T.ToPILImage()
                        image = to_pil(image)
                        
                        # Calculate lacunarity
                        transform = data_transforms["test"]
                        tensor = transform(image)
                        

                        base_lacunarity = Base_Lacunarity(kernel=None)

                        dbc_lacunarity = DBC_Lacunarity(window_size = tuple(tensor.squeeze(0).size()))
                        dbc_lacunarity_value = dbc_lacunarity(tensor)
                        base_lacunarity_value = base_lacunarity(tensor)
                        
                        # Append the result to the list
                        results.append([image_path, base_lacunarity_value.item(), dbc_lacunarity_value.item()])

# Write the results to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Path", "Base_Lacunarity", "DBC_Lacunarity"])
    writer.writerows(results)

print("Lacunarity calculation complete and results saved to CSV file.")


