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
import pdb

from Base_Lacunarity import Base_Lacunarity
from DBC_Lacunarity import DBC_Lacunarity

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Paths
output_main_folder_path = 'Datasets/Lung Cells SEM Images_group1_DC_NEW'

# Data transforms
data_transforms = {
    'transform': transforms.Compose([
        # transforms.Resize((224, 224)),
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
        sta = sta / sta.sum(dim=(1, 2))
        q_levels_h = q_levels.expand(N, self.level_num, self.level_num)
        q_levels_w = q_levels_h.permute(0, 2, 1)
        
        output = torch.stack([q_levels_h, q_levels_w, sta], dim=1)
        output = output.flatten(2).squeeze(0)
        
        return output, quant.squeeze(0), co_occurrence.squeeze(0)

        

def fractal_dimension(image: np.ndarray) -> np.float64:
    """ Calculates the fractal dimension of an image represented by a 2D numpy array.

    The algorithm is a modified box-counting algorithm as described by Wen-Li Lee and Kai-Sheng Hsieh.

    Args:
        image: A 2D array containing a grayscale image. Format should be equivalent to cv2.imread(flags=0).
               The size of the image has no constraints, but it needs to be square (m×m array).
    Returns:
        D: The fractal dimension Df, as estimated by the modified box-counting algorithm.
    """
    M = image.shape[0]  # image shape
    G_min = image.min()  # lowest gray level (0=white)
    G_max = image.max()  # highest gray level (255=black)
    G = G_max - G_min + 1  # number of gray levels, typically 256
    prev = -1  # used to check for plateaus
    r_Nr = []

    for L in range(2, (M // 2) + 1):
        h = max(1, G // (M // L))  # minimum box height is 1
        N_r = 0
        r = L / M
        for i in range(0, M, L):
            boxes = [[]] * ((G + h - 1) // h)  # create enough boxes with height h to fill the fractal space
            for row in image[i:i + L]:  # boxes that exceed bounds are shrunk to fit
                for pixel in row[i:i + L]:
                    height = (pixel - G_min) // h  # lowest box is at G_min and each is h gray levels tall
                    boxes[height].append(pixel)  # assign the pixel intensity to the correct box
            stddev = np.sqrt(np.var(boxes, axis=1))  # calculate the standard deviation of each box
            stddev = stddev[~np.isnan(stddev)]  # remove boxes with NaN standard deviations (empty)
            nBox_r = 2 * (stddev // h) + 1
            N_r += sum(nBox_r)
        if N_r != prev:  # check for plateauing
            r_Nr.append([r, N_r])
            prev = N_r
    x = np.array([np.log(1 / point[0]) for point in r_Nr])  # log(1/r)
    y = np.array([np.log(point[1]) for point in r_Nr])  # log(Nr)
    D = np.polyfit(x, y, 1)[0]  # D = lim r -> 0 log(Nr)/log(1/r)
    return D

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
def calculate_emd(qco_output1, qco_output2):
    # Ensure the count is the first column, then the bin centers
    qco_output1 = qco_output1.transpose(0, 1) 
    qco_output2 = qco_output2.transpose(0, 1)
    qco_output1_swapped = torch.cat((qco_output1[:, 2:], qco_output1[:, :2]), dim=1)
    qco_output2_swapped = torch.cat((qco_output2[:, 2:], qco_output2[:, :2]), dim=1)
    
    qco_output1_np = qco_output1_swapped.cpu().numpy().astype(np.float32)
    qco_output2_np = qco_output2_swapped.cpu().numpy().astype(np.float32)
    
    emd_score, _, _ = cv2.EMD(qco_output1_np, qco_output2_np, cv2.DIST_L2)
    
    return emd_score

def calculate_emd_matrix(class_histograms):
    classes = list(class_histograms.keys())
    n_classes = len(classes)
    emd_matrix = np.zeros((n_classes, n_classes))
    
    for i, class1 in enumerate(classes):
        for j, class2 in enumerate(classes):
            if i != j:
                hist1 = class_histograms[class1]
                hist2 = class_histograms[class2]
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

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_average_lacunarity(average_lacunarity_per_class):
    n_classes = len(average_lacunarity_per_class)
    fig, axes = plt.subplots(1, n_classes + 1, figsize=(20, 4), 
                             gridspec_kw={'width_ratios': [1]*n_classes + [0.05]})
    fig.suptitle('Average Lacunarity Feature Maps by Class')

    # Determine global min and max for consistent color scaling
    all_values = torch.cat(list(average_lacunarity_per_class.values()))
    vmin, vmax = all_values.min().item(), all_values.max().item()

    for idx, (class_name, avg_lacunarity) in enumerate(average_lacunarity_per_class.items()):
        im = axes[idx].imshow(avg_lacunarity.squeeze(0).cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[idx].set_title(class_name)
        axes[idx].axis('off')

    # Add a colorbar to the right of the last image
    cbar = fig.colorbar(im, cax=axes[-1])
    cbar.set_label('Lacunarity Value')

    plt.tight_layout()
    plt.show()



def visualize_representative_lacunarity(representative_lacunarity):
    n_classes = len(representative_lacunarity)
    fig, axes = plt.subplots(1, n_classes + 1, figsize=(20, 4), 
                             gridspec_kw={'width_ratios': [1]*n_classes + [0.05]})
    fig.suptitle('Representative Lacunarity Feature Maps by Class')

    # Determine global min and max for consistent color scaling
    all_values = torch.cat([map.flatten() for map in representative_lacunarity.values()])
    vmin, vmax = all_values.min().item(), all_values.max().item()

    for idx, (class_name, rep_lacunarity) in enumerate(representative_lacunarity.items()):
        im = axes[idx].imshow(rep_lacunarity.squeeze().cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[idx].set_title(class_name)
        axes[idx].axis('off')

    # Add a colorbar to the right of the last image
    cbar = fig.colorbar(im, cax=axes[-1])
    cbar.set_label('Lacunarity Value')

    plt.tight_layout()
    plt.show()

import torch
import torch.nn.functional as F

def aggregate_lacunarity_maps(maps):
    # Normalize each map by subtracting the mean and dividing by the standard deviation
    normalized_maps = [(m - m.mean()) / (m.std() + 1e-8) for m in maps]

    # Flatten the normalized maps for similarity calculation
    flat_maps = torch.stack([m.view(-1) for m in normalized_maps])

    # Calculate the cosine similarity matrix for all pairs
    norm_flat_maps = F.normalize(flat_maps, p=2, dim=1)  # Normalize vectors to unit length (L2 norm)
    sim_matrix = torch.mm(norm_flat_maps, norm_flat_maps.t())  # Matrix multiplication for cosine similarity

    # Compute the weights as the mean of the similarity matrix along the rows
    weights = sim_matrix.mean(dim=1)

    # Weighted aggregation
    weighted_sum = sum(map * weight for map, weight in zip(normalized_maps, weights))
    
    return weighted_sum / weights.sum()



def main():
    # Initialize models
    base_lacunarity = Base_Lacunarity(kernel=(21,21), stride=(1,1)).to(device)
    qco_2d = QCO_2d(scale=1, level_num=5).to(device)

    # Initialize dataset and dataloader
    dataset = LungCells(root=output_main_folder_path, train=True, transform=data_transforms['transform'])
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize storage
    class_lacunarity_maps = defaultdict(list)

    # Process batches
    for images, labels in tqdm(train_loader, desc="Processing images"):
        images = images.to(device)
        base_values = base_lacunarity(images)

        for i in range(len(labels)):
            class_name = dataset.classes[labels[i]]
            lacunarity_map = base_values[i]
            class_lacunarity_maps[class_name].append(lacunarity_map)

    # Aggregate lacunarity maps for each class
    aggregated_lacunarity = {}
    for class_name, lacunarity_maps in class_lacunarity_maps.items():
        aggregated_lacunarity[class_name] = aggregate_lacunarity_maps(lacunarity_maps)

    # Visualize aggregated lacunarity maps
    visualize_representative_lacunarity(aggregated_lacunarity)

    # Pass aggregated lacunarity maps through QCO
    class_qco_outputs = {}
    for class_name, agg_map in aggregated_lacunarity.items():
        output, _, _ = qco_2d(agg_map)
        class_qco_outputs[class_name] = output

    # Calculate and visualize EMD matrix
    emd_matrix, class_names = calculate_emd_matrix(class_qco_outputs)
    print("EMD Matrix:")
    print(pd.DataFrame(emd_matrix, index=class_names, columns=class_names))
    visualize_emd_matrix(emd_matrix, class_names)

if __name__ == "__main__":
    main()
