# -*- coding: utf-8 -*-
"""
Return index of built in Pytorch datasets 
"""
import numpy as np
from torch.utils.data import Dataset
import torch
import ntpath
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import agml
from agml.utils.io import nested_file_list
from sklearn import preprocessing
import torchvision.transforms as T
import pdb
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

#This is for getting all images in a directory (including subdirs)
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
    def __init__(self, root, train = True, transform=None, label_cutoff=1024, load_all=True):
        self.load_all = load_all
        self.transform = transform
        self.label_cutoff = label_cutoff
        self.data = []
        self.targets = []
        self.files = []
        self.classes =  ['Lung Cells Exposed to Crystalline Silica (CS)', 'Lung Cells Exposed to Isocyanate (IPDI)', 
                        'Lung Cells Exposed to Nickel Oxide (NiO)', 'Lung Cells Exposed to Silver Nanoparticles (Ag-NP)',
                        'Lung Cells Untreated']
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
                self._image_files = getListOfFiles(os.path.join(root, "test"))
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
        #Convert to numpy array and normalize to be [0,255]
        image = np.array(image)
        image = (image/image.max()) * 255  
        #Remove label
        image = (image[0:self.label_cutoff,:])
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