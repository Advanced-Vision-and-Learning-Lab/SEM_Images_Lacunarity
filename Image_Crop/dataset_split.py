import os
import shutil
import random

# Paths to the original dataset and the new train/test directories
original_dataset_path = 'Datasets/Lung Cells SEM Images_group1_DC_NEW'
train_dataset_path = 'Datasets/Lung Cells SEM Images_group1_DC_NEW/train'
test_dataset_path = 'Datasets/Lung Cells SEM Images_group1_DC_NEW/test'

# Split ratio
train_ratio = 0.8  # 80% for training, 20% for testing

# Create train and test directories if they do not exist
os.makedirs(train_dataset_path, exist_ok=True)
os.makedirs(test_dataset_path, exist_ok=True)

# Loop through each subfolder (class) in the original dataset
for subfolder in os.listdir(original_dataset_path):
    subfolder_path = os.path.join(original_dataset_path, subfolder)
    if os.path.isdir(subfolder_path):
        # Paths for the class subfolders in train and test directories
        train_subfolder_path = os.path.join(train_dataset_path, subfolder)
        test_subfolder_path = os.path.join(test_dataset_path, subfolder)
        
        # Create class subfolders in train and test directories
        os.makedirs(train_subfolder_path, exist_ok=True)
        os.makedirs(test_subfolder_path, exist_ok=True)
        
        # Get all image files in the class subfolder
        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif'))]
        
        # Shuffle the image files
        random.shuffle(image_files)
        
        # Split the image files into train and test sets
        split_index = int(len(image_files) * train_ratio)
        train_files = image_files[:split_index]
        test_files = image_files[split_index:]
        
        # Copy the image files to the respective train and test directories
        for file_name in train_files:
            src_file = os.path.join(subfolder_path, file_name)
            dest_file = os.path.join(train_subfolder_path, file_name)
            shutil.copy2(src_file, dest_file)
        
        for file_name in test_files:
            src_file = os.path.join(subfolder_path, file_name)
            dest_file = os.path.join(test_subfolder_path, file_name)
            shutil.copy2(src_file, dest_file)

print("Dataset split into train and test sets successfully.")
