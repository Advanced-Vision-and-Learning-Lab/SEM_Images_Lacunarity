import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn import preprocessing
from tqdm import tqdm
from skimage.segmentation import slic
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import ntpath
from Base_Lacunarity import Base_Lacunarity
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize
import torchvision.transforms as T
from barbar import Bar

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.379, std=0.224)
    ]),
    'val': transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.379, std=0.224)
    ]),
}

def Compute_Mean_STD(trainloader):
    print('Computing Mean/STD')
    nimages = 0
    mean = 0.0
    var = 0.0
    for i_batch, batch_target in enumerate(Bar(trainloader)):
        batch = batch_target[0]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        nimages += batch.size(0)
        mean += batch.mean(2).sum(0)
        var += batch.var(2).sum(0)
   
    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)
    print()
    
    return mean, std

# This is for getting all images in a directory (including subdirs)
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
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.join(root, split)
        self.transform = transform
        self.data = []
        self.targets = []
        self.classes = []
        self.class_to_idx = {}

        # Traverse the directory structure
        for class_folder in os.listdir(self.root):
            class_path = os.path.join(self.root, class_folder)
            if os.path.isdir(class_path):
                class_name = class_folder.split('Lung Cells Exposed to ')[-1].split('Lung Cells ')[-1]
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.classes)
                    self.classes.append(class_name)

                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data.append(img_path)
                        self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        target = self.targets[idx]

        image = Image.open(img_path)  # Convert to grayscale
        image = np.array(image)
        image = (image / image.max()) * 255


        if self.transform:
            image = image.astype(np.uint8)
            to_pil = transforms.ToPILImage()
            image = to_pil(image)
            image = self.transform(image)

        return image, target


def visualize_images(dataset, class_name, num_images=5):
    class_idx = dataset.class_to_idx[class_name]
    count = 0
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    fig.suptitle(f'Sample Images from {class_name} Class', fontsize=16)
    
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        if label == class_idx:
            axes[count].imshow(image.squeeze(), cmap='gray')
            axes[count].set_title(f'Image {count+1}')
            axes[count].axis('off')
            count += 1
            if count == num_images:
                break

    plt.show()

def check_image_statistics(dataset, class_name):
    class_idx = dataset.class_to_idx[class_name]
    means = []
    stds = []
    for idx in tqdm(range(len(dataset)), desc=f'Processing {class_name} images'):
        image, label = dataset[idx]
        if label == class_idx:
            image_np = image.squeeze().numpy()
            means.append(np.mean(image_np))
            stds.append(np.std(image_np))
    
    print(f'\n{class_name} Class Image Statistics:')
    print(f'Mean of means: {np.mean(means):.4f}')
    print(f'Mean of stds: {np.mean(stds):.4f}')
    print(f'Min of means: {np.min(means):.4f}')
    print(f'Max of means: {np.max(means):.4f}')
    print(f'Min of stds: {np.min(stds):.4f}')
    print(f'Max of stds: {np.max(stds):.4f}')


def generate_grid(img, numSP):
    h, w = img.shape
    SP_mask = np.zeros((h, w))
    ratio = h / w
    w_num = int(np.ceil(np.sqrt(numSP / ratio)))
    h_num = int(np.ceil(ratio * np.sqrt(numSP / ratio)))
    w_int = np.ceil(w / w_num)
    h_int = np.ceil(h / h_num)
    label = 1
    
    for j in range(h_num):
        for i in range(w_num):
            h_start, h_end = int(j * h_int), int((j + 1) * h_int)
            w_start, w_end = int(i * w_int), int((i + 1) * w_int)
            h_end = min(h_end, h)
            w_end = min(w_end, w)
            SP_mask[h_start:h_end, w_start:w_end] = label
            label += 1
    
    return SP_mask.astype(int)

def calculate_superpixel_lacunarity(image, mask, label):
    superpixel = np.where(mask == label, image, 0)
    if np.all(superpixel == 0):
        print(f"Superpixel {label} is empty")
        return 0
    base_lacunarity = Base_Lacunarity(kernel=None)
    try:
        result = base_lacunarity(torch.from_numpy(superpixel).unsqueeze(0).unsqueeze(0).float()).item()
        if result == 0:
            print(f"Zero lacunarity for non-empty superpixel {label}")
        return result
    except Exception as e:
        print(f"Error calculating lacunarity for superpixel {label}: {e}")
        return 0

def generate_sp_profile(image, numSP=200):
    SP_mask = generate_grid(image, numSP)
    SP_profile = []
    
    for label in range(1, numSP + 1):
        if label in SP_mask:
            lacunarity = calculate_superpixel_lacunarity(image, SP_mask, label)
            cy, cx = np.mean(np.where(SP_mask == label), axis=1)
            SP_profile.append([lacunarity, cx, cy])
    
    SP_profile = np.array(SP_profile)
    
    return {'SP_profile': SP_profile, 'SP_mask': SP_mask, 'Root_mask': SP_mask, 'Img': image}

def process_lung_cell_images(data_dir, num_superpixels=200):
    dataset = LungCells(root=data_dir, transform=data_transforms['train'])
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    # mean, std = Compute_Mean_STD(trainloader=DataLoader)


    class_signatures = {i: [] for i in range(len(dataset.classes))}
    class_image_counts = {i: 0 for i in range(len(dataset.classes))}
    
    for image, label in tqdm(dataloader, desc="Processing images"):
        image_np = image.squeeze().numpy()
        sp_profile = generate_sp_profile(image_np, num_superpixels)
        class_signatures[label.item()].append(sp_profile['SP_profile'])
        class_image_counts[label.item()] += 1
    
    avg_class_signatures = {}
    for label, signatures in class_signatures.items():
        if signatures:
            avg_signature = np.mean(signatures, axis=0)
            avg_class_signatures[label] = avg_signature
        else:
            print(f"No signatures found for class {dataset.classes[label]}")
    
    return avg_class_signatures, class_image_counts, dataset.classes

def visualize_and_save_signature(signature, label, save_dir, num_superpixels=200):
    plt.figure(figsize=(10, 8))
    
    # Calculate grid dimensions
    grid_height = int(np.ceil(np.sqrt(num_superpixels)))
    grid_width = int(np.ceil(num_superpixels / grid_height))
    
    # Create grid and fill with lacunarity values
    grid = np.zeros((grid_height, grid_width))
    for i, (lacunarity, _, _) in enumerate(signature):
        row = i // grid_width
        col = i % grid_width
        grid[row, col] = lacunarity
    
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Average Lacunarity')
    plt.title(f'Average Lacunarity Signature for Class {label}')
    plt.xlabel('Superpixel X')
    plt.ylabel('Superpixel Y')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'average_signature_class_{label}.png'))
    plt.close()



if __name__ == "__main__":
    output_main_folder_path = 'Datasets/Lung_Cells_DC_Split_overlap'
    num_superpixels = 625    
    avg_class_signatures, class_image_counts, class_names = process_lung_cell_images(output_main_folder_path, num_superpixels)
    
    
    signature_dir = 'signatures'
    os.makedirs(signature_dir, exist_ok=True)
    
    for label, avg_sig in avg_class_signatures.items():
        visualize_and_save_signature(avg_sig, class_names[label], signature_dir, num_superpixels)
        avg_lacunarity = np.mean(avg_sig[:, 0])
        print(f"\nClass: {class_names[label]}")
        print(f"  Number of images: {class_image_counts[label]}")
        print(f"  Average lacunarity: {avg_lacunarity:.4f}")
        print(f"  Min lacunarity: {np.min(avg_sig[:, 0]):.4f}")
        print(f"  Max lacunarity: {np.max(avg_sig[:, 0]):.4f}")
        
        if np.all(avg_sig[:, 0] == avg_sig[0, 0]):
            print(f"  WARNING: All superpixels have the same lacunarity value for {class_names[label]}")
    
    print(f"\nAverage signature visualizations saved in the '{signature_dir}' folder.")