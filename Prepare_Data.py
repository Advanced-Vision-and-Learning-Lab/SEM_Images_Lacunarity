# -*- coding: utf-8 -*-
"""
Create datasets and dataloaders for models
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import pdb
from Datasets.Split_Data import DataSplit
import ssl
## PyTorch dependencies
import torch
## Local external libraries
from Datasets.Pytorch_Datasets import *
from Datasets.Get_transform import *



def Prepare_DataLoaders(Network_parameters, split):
    ssl._create_default_https_context = ssl._create_unverified_context
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']    
    global data_transforms
    data_transforms = get_transform(Network_parameters, input_size=224)


    if Dataset == "LungCells_DC" or Dataset == 'LungCells_ME':
        train_dataset = LungCells(root=data_dir, train=True, transform=data_transforms)

        dataloaders_dict = {torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=Network_parameters['batch_size'],
                                                        num_workers=Network_parameters['num_workers'],
                                                        pin_memory=Network_parameters['pin_memory'],
                                                        shuffle=False,
                                                        )}   
      


    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset)) 

    return dataloaders_dict