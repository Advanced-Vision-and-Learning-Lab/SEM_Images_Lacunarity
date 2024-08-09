import os
import csv
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms as T
from sklearn import preprocessing
from Base_Lacunarity import Base_Lacunarity
from DBC_Lacunarity import DBC_Lacunarity
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde
import pdb
import ntpath
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
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
csv_file_path = 'C:/Users/aksha/Peeples_Lab/SEM_Images_Lacunarity/Datasets/DC_changed_lacunarity_results_all.csv'

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.2874, std=0.1090)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.2874, std=0.1090)
    ]),
}


class QCO_2d(nn.Module):
    def __init__(self, scale, level_num):
        super(QCO_2d, self).__init__()
        self.level_num = level_num
        self.scale = scale

    def forward(self, x):
        # Input: x = torch.tensor([[[0.1, 0.4],
        #                           [0.7, 1.0]]])  # Shape: (1, 2, 2)
        N, H, W = x.shape  # N=1, H=2, W=2
        
        cos_sim_min = x.min()  # 0.1
        cos_sim_max = x.max()  # 1.0

        #TRY to make q levels the same across the image
        q_levels = torch.linspace(cos_sim_min, cos_sim_max, self.level_num).to(x.device)
        # q_levels = tensor([0.1, 0.55, 1.0])  # Assuming self.level_num = 3
        q_levels = q_levels.view(1, 1, -1)  # Shape: (1, 1, 3)
        
        x_reshaped = x.view(N, 1, H*W)  # Shape: (1, 1, 4)
        
        sigma = 1 / (self.level_num / 2)  # 2/3
        quant = torch.exp(-(x_reshaped.unsqueeze(-1) - q_levels)**2 / (sigma**2))
        # quant shape: (1, 1, 4, 3)
        
        quant = quant.view(N, H, W, self.level_num)  # Shape: (1, 2, 2, 3)
        quant = quant.permute(0, 3, 1, 2)  # Shape: (1, 3, 2, 2)
        
        quant = F.pad(quant, (0, 1, 0, 1), mode='constant', value=0.)
        # quant shape: (1, 3, 3, 3)
        
        quant_left = quant[:, :, :H, :W].unsqueeze(2)  # Shape: (1, 3, 1, 2, 2)
        quant_right = quant[:, :, 1:, 1:].unsqueeze(1)  # Shape: (1, 1, 3, 2, 2)
        
        co_occurrence = quant_left * quant_right  # Shape: (1, 3, 3, 2, 2)
        
        sta = co_occurrence.sum(dim=(-1, -2))  # Shape: (1, 3, 3)
        
        sta = sta / sta.sum(dim=(1, 2), keepdim=True)
        
        q_levels_h = q_levels.expand(N, self.level_num, self.level_num)  # Shape: (1, 3, 3)
        q_levels_w = q_levels_h.permute(0, 2, 1)  # Shape: (1, 3, 3)
        
        output = torch.stack([q_levels_h, q_levels_w, sta], dim=1)  # Shape: (1, 3, 3, 3)
        
        return output, quant.squeeze(0), co_occurrence.squeeze(0)


# This is for getting all images in a directory (including subdirs)
def getListOfFiles(dirName):
    # create a list of all files in a root dir
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


class GDCB(nn.Module):
    def __init__(self,mfs_dim=25,nlv_bcd=6):
        super(GDCB,self).__init__()
        self.mfs_dim=mfs_dim
        self.nlv_bcd=nlv_bcd
        self.pool=nn.ModuleList()
        
        for i in range(self.nlv_bcd-1):
            self.pool.add_module(str(i),nn.MaxPool2d(kernel_size=i+2,stride=(i+2)//2))
        self.ReLU = nn.ReLU()
        
    def forward(self,input):
        tmp=[]
        for i in range(self.nlv_bcd-1):
            output_item=self.pool[i](input)
            tmp.append(torch.sum(torch.sum(output_item,dim=2,keepdim=True),dim=3,keepdim=True))
        output=torch.cat(tuple(tmp),2)
        output=torch.log2(self.ReLU(output)+1)
        X=[-math.log(i+2,2) for i in range(self.nlv_bcd-1)]
        X = torch.tensor(X).to(output.device)
        X=X.view([1,1,X.shape[0],1])
        meanX = torch.mean(X,2,True)
        meanY = torch.mean(output,2,True)
        Fracdim = torch.div(torch.sum((output-meanY)*(X-meanX),2,True),torch.sum((X-meanX)**2,2,True))
        return Fracdim
    

    
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
        # Convert to numpy array and normalize to be [0, 255]
        image = np.array(image)
        image = (image/image.max()) * 255
        target = datafiles["label"]

        if self.transform:
            image = image.astype(np.uint8)
            to_pil = T.ToPILImage()
            image = to_pil(image)
            image = self.transform(image)
        return image, target


# List to store the results
results = []

train_dataset = LungCells(root=output_main_folder_path, train=True, transform=data_transforms['train'])
val_dataset = LungCells(root=output_main_folder_path, train=False, transform=data_transforms['val'])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Initialize lacunarity models
base_lacunarity = Base_Lacunarity(kernel=(3,3), stride=(1,1))
dbc_lacunarity = DBC_Lacunarity(window_size=3)


# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
base_lacunarity.to(device)
dbc_lacunarity.to(device)

class_lacunarity = defaultdict(lambda: defaultdict(list))
class_sta_sums = defaultdict(lambda: torch.zeros(5, 5).to(device))
class_sta_counts = defaultdict(int)


class_histograms = defaultdict(list)

def process_batch(images, labels, dataset):
    global class_sta_sums, class_sta_counts, class_histograms
    images = images.to(device)
    
    base_values = base_lacunarity(images)

    for i in range(len(labels)):
        class_name = dataset.classes[labels[i]]
        for lac_type, lac_values in [('Base', base_values[i])]:
            output, quant, co_occurrence = qco_2d(lac_values)

            # Extract sta from output
            sta = output[0, 2]  # This is the 2D histogram we want

            # Accumulate sta for each class
            class_sta_sums[class_name] += sta
            class_sta_counts[class_name] += 1
            
            # Store the 2D histogram for this class
            class_histograms[class_name].append(sta)

        results.append([
            class_name, 
            torch.mean(base_values[i]).item(), 
        ])


def calculate_emd(hist1, hist2):
    hist1_np = hist1.cpu().numpy().astype(np.float32)
    hist2_np = hist2.cpu().numpy().astype(np.float32)
    
    # Create coordinate arrays for 2D histogram
    h, w = hist1_np.shape
    h1, w1 = hist2_np.shape
    #coords are dependent on quantization level
    coords = np.array([(i, j) for i in range(h) for j in range(w)], dtype=np.float32)
    
    # Flatten the 2D histograms
    hist1_flat = hist1_np.reshape(-1)
    hist2_flat = hist2_np.reshape(-1)
    
    # Calculate EMD
    emd_score, _, _ = cv2.EMD(
        np.column_stack((hist1_flat, coords)), 
        np.column_stack((hist2_flat, coords)), 
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
    # Prepare data for plotting
    data = []
    for class_name, sta_avg in class_sta_avgs.items():
        sta_np = sta_avg.cpu().numpy().flatten()
        data.extend([(class_name, val) for val in sta_np])
    
    df = pd.DataFrame(data, columns=['Class', 'Statistical Texture Value'])

    # 1. Box plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Class', y='Statistical Texture Value', data=df)
    plt.title('Distribution of Average Statistical Texture Values by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Class', y='Statistical Texture Value', data=df)
    plt.title('Distribution of Average Statistical Texture Values by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. 2D histogram plot with consistent color scale and vertical colorbar
    n_classes = len(class_sta_avgs)
    fig, axes = plt.subplots(1, n_classes + 1, figsize=(20 + 2, 4),
                             gridspec_kw={'width_ratios': [1] * n_classes + [0.1]})
    
    # Determine global min and max values
    global_min = min(sta_avg.min().item() for sta_avg in class_sta_avgs.values())
    global_max = max(sta_avg.max().item() for sta_avg in class_sta_avgs.values())

    for i, (class_name, sta_avg) in enumerate(class_sta_avgs.items()):
        im = sns.heatmap(sta_avg.cpu().numpy(), ax=axes[i], cmap='viridis',
                         vmin=global_min, vmax=global_max, cbar=False)
        axes[i].set_title(class_name)
        axes[i].axis('off')
    
    # Add a colorbar to the right of the last plot
    fig.colorbar(im.collections[0], cax=axes[-1], orientation='vertical')
    axes[-1].set_ylabel('Statistical Texture Value')
    
    plt.tight_layout()
    plt.show()



    # 5. Histogram and KDE plot
    plt.figure(figsize=(12, 7))
    for class_name, sta_avg in class_sta_avgs.items():
        data = sta_avg.cpu().numpy().flatten()
        
        # Plot histogram
        plt.hist(data, bins=10, alpha=0.3, density=True, label=f'{class_name} (Histogram)')
        
        # Calculate and plot KDE
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



qco_2d = QCO_2d(scale=1, level_num=5).to(device)

for loader in [train_loader, val_loader]:
    for images, labels in tqdm(loader, desc="Processing images"):
        process_batch(images, labels, loader.dataset)

# Calculate average sta for each class
class_sta_avgs = {class_name: sta_sum / class_sta_counts[class_name] 
                  for class_name, sta_sum in class_sta_sums.items()}

# Visualize the class distributions
visualize_class_sta_distributions(class_sta_avgs)

# After processing all images
emd_matrix, class_names = calculate_emd_matrix(class_histograms)

# Print the EMD matrix
print("EMD Matrix:")
print(pd.DataFrame(emd_matrix, index=class_names, columns=class_names))

# Visualize the EMD matrix
visualize_emd_matrix(emd_matrix, class_names)

# Save results to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Class", "Base_Lacunarity"])
    writer.writerows(results)

print("Lacunarity calculation complete and results saved to CSV file.")