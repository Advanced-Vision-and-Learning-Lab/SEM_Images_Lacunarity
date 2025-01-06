# -*- coding: utf-8 -*-
"""
Create datasets and dataloaders for models
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import pdb
import ssl
## PyTorch dependencies
import torch
## Local external libraries
from Datasets.Dataset_Class import *
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Data transforms
data_transforms = {
    'transform': transforms.Compose([
        transforms.ToTensor(),
    ])
}

def Prepare_DataLoaders(Network_parameters):
    ssl._create_default_https_context = ssl._create_unverified_context
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']

    if Dataset == "LungCells_DC":
        dataset = LungCells(root=data_dir, train=True, transform=data_transforms['transform'])
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)     


    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset)) 

    return dataset, loader