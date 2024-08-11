import os
import csv
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms as T
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from collections import defaultdict
from tqdm import tqdm
import ntpath
import math
import cv2

from Base_Lacunarity import Base_Lacunarity
from DBC_Lacunarity import DBC_Lacunarity

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Paths
output_main_folder_path = 'Datasets/Lung Cells SEM Images_group1_DC_NEW'
csv_file_path = 'C:/Users/aksha/Peeples_Lab/SEM_Images_Lacunarity/Datasets/DC_changed_lacunarity_results_all.csv'

# Data transforms
data_transforms = {
    'transform': transforms.Compose([
        transforms.ToTensor(),
    ])
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Models
class QCO_2d(nn.Module):
    def __init__(self, scale, level_num):
        super(QCO_2d, self).__init__()
        self.level_num = level_num
        self.scale = scale

    def forward(self, x):
        N, H, W = x.shape
        cos_sim_min, cos_sim_max = x.min(), x.max()
        q_levels = torch.linspace(cos_sim_min, cos_sim_max, self.level_num).to(x.device)
        q_levels = q_levels.view(1, 1, -1)
        
        x_reshaped = x.view(N, 1, H*W)
        sigma = 1 / (self.level_num / 2)
        quant = torch.exp(-(x_reshaped.unsqueeze(-1) - q_levels)**2 / (sigma**2))
        
        quant = quant.view(N, H, W, self.level_num).permute(0, 3, 1, 2)
        quant = F.pad(quant, (0, 1, 0, 1), mode='constant', value=0.)
        
        quant_left = quant[:, :, :H, :W].unsqueeze(2)
        quant_right = quant[:, :, 1:, 1:].unsqueeze(1)
        
        co_occurrence = quant_left * quant_right
        sta = co_occurrence.sum(dim=(-1, -2))
        sta = sta / sta.sum(dim=(1, 2), keepdim=True)
        
        q_levels_h = q_levels.expand(N, self.level_num, self.level_num)
        q_levels_w = q_levels_h.permute(0, 2, 1)
        
        output = torch.stack([q_levels_h, q_levels_w, sta], dim=1)
        
        return output, quant.squeeze(0), co_occurrence.squeeze(0)

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

# Dataset
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
        
        if self.load_all:
            self._image_files = self.getListOfFiles(os.path.join(root))
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

    @staticmethod
    def getListOfFiles(dirName):
        listOfFile = os.listdir(dirName)
        allFiles = []
        for entry in listOfFile:
            fullPath = os.path.join(dirName, entry)
            if os.path.isdir(fullPath):
                allFiles = allFiles + LungCells.getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)
        return allFiles

# Helper functions
def calculate_emd(hist1, hist2):
    hist1_np = hist1.cpu().numpy().astype(np.float32)
    hist2_np = hist2.cpu().numpy().astype(np.float32)
    
    min_val, max_val = hist1_np.min(), hist1_np.max()
    min_val1, max_val1 = hist2_np.min(), hist2_np.max()
    
    h, w = hist1_np.shape
    h1, w1 = hist2_np.shape
    
    x_centers = np.linspace(min_val, max_val, h)
    y_centers = np.linspace(min_val, max_val, w)
    x_centers1 = np.linspace(min_val1, max_val1, h1)
    y_centers1 = np.linspace(min_val1, max_val1, w1)
    
    coords = np.array([(x, y) for x in x_centers for y in y_centers], dtype=np.float32)
    coords1 = np.array([(x, y) for x in x_centers1 for y in y_centers1], dtype=np.float32)
    
    hist1_flat = hist1_np.reshape(-1)
    hist2_flat = hist2_np.reshape(-1)
    
    emd_score, _, _ = cv2.EMD(
        np.column_stack((hist1_flat, coords)), 
        np.column_stack((hist2_flat, coords1)), 
        cv2.DIST_L2
    )
    return emd_score

def calculate_emd_matrix(class_histograms):
    classes = list(class_histograms.keys())
    n_classes = len(classes)
    emd_matrix = np.zeros((n_classes, n_classes))
    
    for i, class1 in enumerate(classes):
        for j, class2 in enumerate(classes):
            if i != j:
                hist1 = torch.stack(class_histograms[class1]).mean(dim=0)
                hist2 = torch.stack(class_histograms[class2]).mean(dim=0)
                emd_matrix[i, j] = calculate_emd(hist1, hist2)
    return emd_matrix, classes

def visualize_class_sta_distributions(class_sta_avgs):
    data = []
    for class_name, sta_avg in class_sta_avgs.items():
        sta_np = sta_avg.cpu().numpy().flatten()
        data.extend([(class_name, val) for val in sta_np])
    
    df = pd.DataFrame(data, columns=['Class', 'Statistical Texture Value'])

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Class', y='Statistical Texture Value', data=df)
    plt.title('Distribution of Average Statistical Texture Values by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Class', y='Statistical Texture Value', data=df)
    plt.title('Distribution of Average Statistical Texture Values by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    n_classes = len(class_sta_avgs)
    fig, axes = plt.subplots(1, n_classes + 1, figsize=(20 + 2, 4),
                             gridspec_kw={'width_ratios': [1] * n_classes + [0.1]})
    
    global_min = min(sta_avg.min().item() for sta_avg in class_sta_avgs.values())
    global_max = max(sta_avg.max().item() for sta_avg in class_sta_avgs.values())

    for i, (class_name, sta_avg) in enumerate(class_sta_avgs.items()):
        im = sns.heatmap(sta_avg.cpu().numpy(), ax=axes[i], cmap='viridis',
                         vmin=global_min, vmax=global_max, cbar=False)
        axes[i].set_title(class_name)
        axes[i].axis('off')
    
    fig.colorbar(im.collections[0], cax=axes[-1], orientation='vertical')
    axes[-1].set_ylabel('Statistical Texture Value')
    
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 7))
    for class_name, sta_avg in class_sta_avgs.items():
        data = sta_avg.cpu().numpy().flatten()
        plt.hist(data, bins=10, alpha=0.3, density=True, label=f'{class_name} (Histogram)')
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        plt.plot(x_range, kde(x_range), label=f'{class_name} (KDE)')

    plt.title('Distribution of Average Statistical Texture Values by Class (Histogram and KDE)')
    plt.xlabel('Statistical Texture Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_emd_matrix(emd_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(emd_matrix, annot=True, fmt=".4f", cmap="YlGnBu", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Earth Mover's Distance (EMD) between Classes")
    plt.tight_layout()
    plt.show()



# Main execution
def main():
    # Initialize models
    base_lacunarity = Base_Lacunarity(kernel=(21,21), stride=(1,1)).to(device)
    qco_2d = QCO_2d(scale=1, level_num=3).to(device)

    # Initialize dataset and dataloader
    dataset = LungCells(root=output_main_folder_path, train=True, transform=data_transforms['transform'])
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize storage
    results = []
    class_lacunarity_values = defaultdict(list)
    class_sta_sums = defaultdict(lambda: torch.zeros(3, 3).to(device))
    class_sta_counts = defaultdict(int)
    class_histograms = defaultdict(list)

    # Process batches
    for images, labels in tqdm(train_loader, desc="Processing images"):
        images = images.to(device)
        base_values = base_lacunarity(images)

        for i in range(len(labels)):
            class_name = dataset.classes[labels[i]]
            lacunarity_value = torch.mean(base_values[i]).item()
            
            class_lacunarity_values[class_name].append(lacunarity_value)
            
            output, _, _ = qco_2d(base_values[i])
            sta = output[0, 2]

            class_sta_sums[class_name] += sta
            class_sta_counts[class_name] += 1
            class_histograms[class_name].append(sta)

            results.append([class_name, lacunarity_value])

    # Calculate and print average local lacunarity
    print("\nAverage Local Lacunarity for each class:")
    for class_name, lacunarity_values in class_lacunarity_values.items():
        avg_lacunarity = np.mean(lacunarity_values)
        print(f"{class_name}: {avg_lacunarity:.4f}")

    # Calculate average sta for each class
    class_sta_avgs = {class_name: sta_sum / class_sta_counts[class_name] 
                      for class_name, sta_sum in class_sta_sums.items()}

    # Visualize the class distributions
    visualize_class_sta_distributions(class_sta_avgs)

    # Calculate and visualize EMD matrix
    emd_matrix, class_names = calculate_emd_matrix(class_histograms)
    print("EMD Matrix:")
    print(pd.DataFrame(emd_matrix, index=class_names, columns=class_names))
    visualize_emd_matrix(emd_matrix, class_names)


if __name__ == "__main__":
    main()