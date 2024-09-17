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

# Dataset
class LungCells(Dataset):
    def __init__(self, root, train=True, transform=None, label_cutoff=1024, load_all=True):
        self.load_all = load_all
        self.transform = transform
        self.label_cutoff = label_cutoff
        self.data = []
        self.targets = []
        self.files = []
        self.classes = ['Crystalline Silica (CS)', 
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