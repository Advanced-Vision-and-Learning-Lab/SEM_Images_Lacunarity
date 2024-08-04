import os
import csv
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms as T
from sklearn import preprocessing
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde
import pdb
import ntpath
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from scipy import stats
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Path to the output main folder where the divided images are saved
output_main_folder_path = 'Datasets/Lung Cells SEM Images_group1_DC_NEW'

# Path to the CSV file where the results will be saved
csv_file_path = 'C:/Users/aksha/Peeples_Lab/SEM_Images_Lacunarity/Datasets/DC_changed_fractal_dimension_results_all.csv'

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}

class GDCB(nn.Module):
    def __init__(self, mfs_dim=25, nlv_bcd=6):
        super(GDCB, self).__init__()
        self.mfs_dim = mfs_dim
        self.nlv_bcd = nlv_bcd
        self.pool = nn.ModuleList()
        
        for i in range(self.nlv_bcd-1):
            self.pool.add_module(str(i), nn.MaxPool2d(kernel_size=i+2, stride=(i+2)//2))
        self.ReLU = nn.ReLU()
        
    def forward(self, input):
        tmp = []
        for i in range(self.nlv_bcd-1):
            output_item = self.pool[i](input)
            tmp.append(torch.sum(torch.sum(output_item, dim=2, keepdim=True), dim=3, keepdim=True))
        output = torch.cat(tuple(tmp), 2)
        output = torch.log2(self.ReLU(output)+1)
        X = [-math.log(i+2, 2) for i in range(self.nlv_bcd-1)]
        X = torch.tensor(X).to(output.device)
        X = X.view([1, 1, X.shape[0], 1])
        meanX = torch.mean(X, 2, True)
        meanY = torch.mean(output, 2, True)
        Fracdim = torch.div(torch.sum((output-meanY)*(X-meanX), 2, True), torch.sum((X-meanX)**2, 2, True))
        return Fracdim

def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

class LungCells(Dataset):
    def __init__(self, root, train=True, transform=None, label_cutoff=1024, load_all=True):
        self.load_all = load_all
        self.transform = transform
        self.label_cutoff = label_cutoff
        self.data = []
        self.targets = []
        self.files = []
        self.classes = ['Silver Nanoparticles (Ag-NP)', 'Crystalline Silica (CS)', 'Isocyanate (IPDI)', 
                        'Nickel Oxide (NiO)', 'Untreated']
        if train:
            if self.load_all:
                self._image_files = getListOfFiles(os.path.join(root))
                for img_name in self._image_files:
                    self.data.append(Image.open(img_name))
                    self.targets.append((ntpath.basename(img_name).split('_')[0]))
                    
            label_encoder = preprocessing.LabelEncoder()
            self.targets = label_encoder.fit_transform(self.targets)

            for item in zip(self.data, self.targets):
                self.files.append({
                        "img": item[0],
                        "label": item[1]
                        })
                
        else:
            if self.load_all:
                self._image_files = getListOfFiles(os.path.join(root))
                for img_name in self._image_files:
                    self.data.append(Image.open(img_name))
                    self.targets.append((ntpath.basename(img_name).split('_')[0]))
                    
            label_encoder = preprocessing.LabelEncoder()
            self.targets = label_encoder.fit_transform(self.targets)

            for item in zip(self.data, self.targets):
                self.files.append({
                        "img": item[0],
                        "label": item[1]
                        })

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):       
        datafiles = self.files[idx]
        image = datafiles["img"]
        image = np.array(image)
        image = (image/image.max()) * 255
        target = datafiles["label"]

        if self.transform:
            image = image.astype(np.uint8)
            to_pil = T.ToPILImage()
            image = to_pil(image)
            image = self.transform(image)
        return image, target

results = []
class_fractal_dims = defaultdict(list)

train_dataset = LungCells(root=output_main_folder_path, train=True, transform=data_transforms['train'])
val_dataset = LungCells(root=output_main_folder_path, train=False, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

gdcb = GDCB(mfs_dim=25, nlv_bcd=6).to(device)

def process_batch(images, labels, dataset):
    global class_fractal_dims
    images = images.to(device)
    
    fractal_dims = gdcb(images)

    for i in range(len(labels)):
        class_name = dataset.classes[labels[i]]
        class_fractal_dims[class_name].append(fractal_dims[i].item())

        results.append([
            class_name, 
            fractal_dims[i].item(), 
        ])

def visualize_fractal_dimensions(class_fractal_dims):
    plt.figure(figsize=(12, 6))
    
    for class_name, dims in class_fractal_dims.items():
        sns.kdeplot(dims, label=class_name)
    
    plt.title('Distribution of Fractal Dimensions by Class')
    plt.xlabel('Fractal Dimension')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Box plot
    df = pd.DataFrame([(class_name, dim) for class_name, dims in class_fractal_dims.items() for dim in dims],
                      columns=['Class', 'Fractal Dimension'])
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Class', y='Fractal Dimension', data=df)
    plt.title('Distribution of Fractal Dimensions by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def calculate_fractal_dim_distance_matrix(class_fractal_dims):
    classes = list(class_fractal_dims.keys())
    n_classes = len(classes)
    distance_matrix = np.zeros((n_classes, n_classes))
    
    for i, class1 in enumerate(classes):
        for j, class2 in enumerate(classes):
            if i != j:
                mean1 = np.mean(class_fractal_dims[class1])
                mean2 = np.mean(class_fractal_dims[class2])
                distance_matrix[i, j] = abs(mean1 - mean2)
    
    return distance_matrix, classes

def visualize_fractal_dim_distance_matrix(distance_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, annot=True, fmt=".4f", cmap="YlGnBu", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Fractal Dimension Distance between Classes")
    plt.tight_layout()
    plt.show()

for loader in [train_loader, val_loader]:
    for images, labels in tqdm(loader, desc="Processing images"):
        process_batch(images, labels, loader.dataset)

# Visualize the fractal dimension distributions
visualize_fractal_dimensions(class_fractal_dims)

# Calculate and visualize the fractal dimension distance matrix
distance_matrix, class_names = calculate_fractal_dim_distance_matrix(class_fractal_dims)

# Print the distance matrix
print("Fractal Dimension Distance Matrix:")
print(pd.DataFrame(distance_matrix, index=class_names, columns=class_names))

# Visualize the distance matrix
visualize_fractal_dim_distance_matrix(distance_matrix, class_names)

# Save results to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Class", "Fractal_Dimension"])
    writer.writerows(results)

print("Fractal dimension calculation complete and results saved to CSV file.")