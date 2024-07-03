import os
import csv
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms as T
from sklearn import preprocessing
from Base_Lacunarity import Base_Lacunarity
from DBC_Lacunarity import DBC_Lacunarity
from Multi_Scale_Lacunarity import MS_Lacunarity
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde
from scipy.stats import entropy
import pdb
import ntpath
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from barbar import Bar
from torch.utils.data import DataLoader
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Path to the output main folder where the divided images are saved
output_main_folder_path = 'Datasets/Lung_Cells_DC_Split_overlap/'

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

def Compute_Mean_STD(trainloader):
    print('Computing Mean/STD')
    nimages = 0
    mean = 0.0
    var = 0.0
    for i_batch, batch_target in enumerate(Bar(trainloader)):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        var += batch.var(2).sum(0)
   
    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)
    print()
    
    return mean, std

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
                self._image_files = getListOfFiles(os.path.join(root, "train"))
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
                self._image_files = getListOfFiles(os.path.join(root, "val"))
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
    
    def display_image(self, idx):
        image, target = self[idx]
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()  # Rearrange tensor to image format
        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {self.classes[target]}')
        plt.axis('off')
        plt.show()

sum_pixels = 0
sum_squared_pixels = 0
num_pixels = 0

# List to store the results
results = []

train_dataset = LungCells(root=output_main_folder_path, train=True, transform=data_transforms['train'])
val_dataset = LungCells(root=output_main_folder_path, train=False, transform=data_transforms['val'])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# Initialize lacunarity models
base_lacunarity = Base_Lacunarity(kernel=(3,3), stride=(1,1))
dbc_lacunarity = DBC_Lacunarity(window_size=3)
multi_lacunarity = MS_Lacunarity(num_levels=3)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
base_lacunarity.to(device)
dbc_lacunarity.to(device)
multi_lacunarity.to(device)

class_lacunarity = defaultdict(lambda: defaultdict(list))
results = []

def process_batch(images, labels, dataset, num_bins=30):
    images = images.to(device)
    
    base_values = base_lacunarity(images)
    dbc_values = dbc_lacunarity(images)
    multi_values = multi_lacunarity(images)

    for i in range(len(labels)):
        class_name = dataset.classes[labels[i]]
        for lac_type, lac_values in [('Base', base_values), ('DBC', dbc_values), ('Multi', multi_values)]:
            values = lac_values[i].cpu().detach().flatten()
            
            # Compute histogram
            bin_edges = torch.linspace(values.min(), values.max(), num_bins + 1)
            hist = torch.histogram(values, bin_edges, density=True)[0]
            
            class_lacunarity[class_name][f'{lac_type}_hist'] += hist
            class_lacunarity[class_name][f'{lac_type}_edges'] = bin_edges
            class_lacunarity[class_name][lac_type].append(values.numpy())

        results.append([
            class_name, 
            torch.mean(base_values[i]).item(), 
            torch.mean(dbc_values[i]).item(),
            torch.mean(multi_values[i]).item()
        ])
# Process train and validation data
for loader in [train_loader, val_loader]:
    for images, labels in tqdm(loader, desc="Processing images"):
        process_batch(images, labels, loader.dataset)

# Initialize histograms
num_bins = 30
for class_name in train_dataset.classes:
    for lac_type in ['Base', 'DBC', 'Multi']:
        class_lacunarity[class_name][f'{lac_type}_hist'] = torch.zeros(num_bins)
        class_lacunarity[class_name][f'{lac_type}_edges'] = None

# Save results to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Class", "Base_Lacunarity", "DBC_Lacunarity", "Multi_Lacunarity"])
    writer.writerows(results)

print("Lacunarity calculation complete and results saved to CSV file.")


# def plot_histograms_with_curves(lacunarity_type):
#     plt.figure(figsize=(12, 8))

#     predefined_class_order = [
#         "Crystalline Silica (CS)", 
#         "Isocyanate (IPDI)", 
#         "Untreated", 
#         "Silver Nanoparticles (Ag-NP)", 
#         "Nickel Oxide (NiO)"
#     ]

#     colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(predefined_class_order)))

#     for class_name, color in zip(predefined_class_order, colors):
#         if class_name in class_lacunarity:
#             hist = class_lacunarity[class_name][f'{lacunarity_type}_hist'].numpy()
#             edges = class_lacunarity[class_name][f'{lacunarity_type}_edges'].numpy()
#             bin_centers = (edges[:-1] + edges[1:]) / 2

#             plt.plot(bin_centers, hist, color=color, linewidth=2, label=class_name)

#             # For KDE curve
#             data = np.concatenate(class_lacunarity[class_name][lacunarity_type])
#             kde = gaussian_kde(data)
#             x_range = np.linspace(data.min(), data.max(), 200)
#             kde_values = kde(x_range)
#             plt.plot(x_range, kde_values, color=color, linestyle='--', linewidth=1)

#     plt.title(f'Distribution of {lacunarity_type} Lacunarity by Class')
#     plt.xlabel('Lacunarity Value')
#     plt.ylabel('Density')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f'{lacunarity_type}_local_lacunarity_distribution.png', dpi=300)
#     plt.close()
    
def plot_histograms_with_curves(lacunarity_type):
    plt.figure(figsize=(12, 8))

    # Define the predefined order for the classes
    predefined_class_order = [
        "Crystalline Silica (CS)", 
        "Isocyanate (IPDI)", 
        "Untreated", 
        "Silver Nanoparticles (Ag-NP)", 
        "Nickel Oxide (NiO)"
    ]

    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(predefined_class_order)))

    for class_name, color in zip(predefined_class_order, colors):
        if class_name in class_lacunarity:
            data = np.concatenate([arr.reshape(-1) for arr in class_lacunarity[class_name][lacunarity_type]])
            data = data[(data >= 0) & (data <= 0.4)]  # Filter values between 0 and 1

            counts, bins, _ = plt.hist(data, bins=30, alpha=0.3, color=color, density=True, label=class_name)

            kde = gaussian_kde(data)
            x_range = np.linspace(min(data), max(data), 200)
            kde_values = kde(x_range)

            plt.plot(x_range, kde_values, color=color, linewidth=2)

    plt.title(f'Distribution of {lacunarity_type} Lacunarity by Class')
    plt.xlabel('Lacunarity Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{lacunarity_type}_local_lacunarity_distribution.png', dpi=300)
    plt.close()

# Plot distributions for each lacunarity type
plot_histograms_with_curves('Base')
plot_histograms_with_curves('DBC')
plot_histograms_with_curves('Multi')


def calculate_histogram_loss(hist1, hist2):
    # Normalize the histograms
    p = hist1 / np.sum(hist1)
    q = hist2 / np.sum(hist2)
    
    # Calculate KL divergence
    kl_div = entropy(p, q)
    
    return kl_div

# Calculate and print histogram loss for each pair of classes
classes = list(class_lacunarity.keys())
for i in range(len(classes)):
    for j in range(i+1, len(classes)):
        class1 = classes[i]
        class2 = classes[j]
        hist1, _ = np.histogram(np.concatenate([arr.reshape(-1) for arr in class_lacunarity[class1]['Base']]), bins=30)
        hist2, _ = np.histogram(np.concatenate([arr.reshape(-1) for arr in class_lacunarity[class2]['Base']]), bins=30)
        loss = calculate_histogram_loss(hist1, hist2)
        print(f"Histogram loss between {class1} and {class2}: {loss}")
