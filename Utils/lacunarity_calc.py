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
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from barbar import Bar
import ntpath
import pdb

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Path to the output main folder where the divided images are saved
output_main_folder_path = 'Datasets/Lung_Cells_DC_Split_overlap/'

# Path to the CSV file where the results will be saved
csv_file_path = 'C:/Users/aksha/Peeples_Lab/SEM_Images_Lacunarity/Datasets/DC_changed_lacunarity_results_all.csv'

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

class_lacunarity_stats = defaultdict(lambda: {"sum": 0, "count": 0})
    
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
base_lacunarity = Base_Lacunarity(kernel=None)

# Move to GPU if available
device = torch.device("cpu")
base_lacunarity.to(device)

class_lacunarity = defaultdict(lambda: defaultdict(list))
results = []

def process_single_image(image, label, dataset):
    base_value = base_lacunarity(image).item()

    class_name = dataset.classes[label]
    class_lacunarity[class_name]['Base'].append(base_value)
    class_lacunarity_stats[class_name]["sum"] += base_value
    class_lacunarity_stats[class_name]["count"] += 1
    results.append([class_name, base_value])

# Process train and validation data
for loader in [train_loader, val_loader]:
    for image, label in tqdm(loader, desc="Processing images"):
        process_single_image(image, label, loader.dataset)

print("\nAverage Lacunarity for each class:")
for class_name, stats in class_lacunarity_stats.items():
    avg_lacunarity = stats["sum"] / stats["count"]
    print(f"{class_name}: {avg_lacunarity:.4f}")

# Save results to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Class", "Base_Lacunarity"])
    writer.writerows(results)

print("Lacunarity calculation complete and results saved to CSV file.")

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
            data = class_lacunarity[class_name][lacunarity_type]
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
    plt.savefig(f'{lacunarity_type}_global_lacunarity_distribution.png', dpi=300)
    plt.close()

# Plot distributions for each lacunarity type
plot_histograms_with_curves('Base')

