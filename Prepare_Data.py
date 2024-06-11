<<<<<<< HEAD
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
from barbar import Bar
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from torch.utils.data import DataLoader, Subset

def Compute_Mean_STD(trainloader):
    print('Computing Mean/STD')
    'Code from: https://stackoverflow.com/questions/60101240/finding-mean-and-standard-deviation-across-image-channels-pytorch'
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

def get_train_test_split(full_dataset, labels, test_size=0.2, random_state=42):
    train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=test_size, stratify=labels, random_state=random_state)
    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)
    return train_dataset, test_dataset


def Prepare_DataLoaders(Network_parameters, split):
    ssl._create_default_https_context = ssl._create_unverified_context
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']    
    global data_transforms
    data_transforms = get_transform(Network_parameters, input_size=224)


    if Dataset == "LungCells":
        full_dataset = LungCells(data_dir, transform=None)
        labels = [full_dataset[i][1] for i in range(len(full_dataset))]

        skf = StratifiedKFold(n_splits=5, shuffle=True)

        for split, (train_index, val_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            train_subset = Subset(full_dataset, train_index)
            test_subset = Subset(full_dataset, val_index)
            
            # Apply transforms to train and validation subsets
            train_subset.dataset.transform = data_transforms["train"]
            test_subset.dataset.transform = data_transforms["test"]

            image_datasets = {'train': train_subset, 'val': test_subset, 'test': test_subset}
            dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=Network_parameters['batch_size'][x],
                                                        num_workers=Network_parameters['num_workers'],
                                                        pin_memory=Network_parameters['pin_memory'],
                                                        shuffle=False,
                                                        )
                                                        for x in ['train', 'val','test']}
                                                        

    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset)) 
    



    if Dataset=='LungCells':
            pass
    
    else:
        image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=Network_parameters['batch_size'][x],
                                                        num_workers=Network_parameters['num_workers'],
                                                        pin_memory=Network_parameters['pin_memory'],
                                                        shuffle=False,
                                                        )
                                                        for x in ['train', 'val','test']}
                                                        


=======
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
from barbar import Bar

def Compute_Mean_STD(trainloader):
    print('Computing Mean/STD')
    'Code from: https://stackoverflow.com/questions/60101240/finding-mean-and-standard-deviation-across-image-channels-pytorch'
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


def Prepare_DataLoaders(Network_parameters, split):
    ssl._create_default_https_context = ssl._create_unverified_context
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']    
    global data_transforms
    data_transforms = get_transform(Network_parameters, input_size=224)


    if Dataset == "LeavesTex":
        train_dataset = LeavesTex1200(data_dir,transform=data_transforms["train"])
        val_dataset = LeavesTex1200(data_dir,transform=data_transforms["test"])
        test_dataset = LeavesTex1200(data_dir,transform=data_transforms["test"])
    
         #Create train/val/test loader
        split = DataSplit(train_dataset,val_dataset,test_dataset, shuffle=False,random_seed=split)
        train_loader, val_loader , test_loader = split.get_split(batch_size=Network_parameters['batch_size']['train'], 
                                                                num_workers=Network_parameters['num_workers'],
                                                                show_sample=False,
                                                                val_batch_size=Network_parameters['batch_size']['val'],
                                                                test_batch_size=Network_parameters['batch_size']['test'])
        dataloaders_dict = {'train': train_loader,'val': val_loader,'test': test_loader}



    
    elif Dataset == "PlantVillage":
        train_dataset = PlantVillage(data_dir,transform=data_transforms["train"])
        val_dataset = PlantVillage(data_dir,transform=data_transforms["test"])
        test_dataset = PlantVillage(data_dir,transform=data_transforms["test"])
    
         #Create train/val/test loader based on mean and std
        split = DataSplit(train_dataset,val_dataset,test_dataset, shuffle=False,random_seed=split)
        train_loader, val_loader , test_loader = split.get_split(batch_size=Network_parameters['batch_size']['train'], 
                                                                num_workers=Network_parameters['num_workers'],
                                                                show_sample=False,
                                                                val_batch_size=Network_parameters['batch_size']['val'],
                                                                test_batch_size=Network_parameters['batch_size']['test'])
        dataloaders_dict = {'train': train_loader,'val': val_loader,'test': test_loader}



    elif Dataset == "DeepWeeds":
        train_dataset = DeepWeeds(data_dir,transform=data_transforms["train"])
        val_dataset = DeepWeeds(data_dir,transform=data_transforms["test"])
        test_dataset = DeepWeeds(data_dir,transform=data_transforms["test"])
    
         #Create train/val/test loader based on mean and std
        split = DataSplit(train_dataset,val_dataset,test_dataset, shuffle=False,random_seed=split)
        train_loader, val_loader , test_loader = split.get_split(batch_size=Network_parameters['batch_size']['train'], 
                                                                num_workers=Network_parameters['num_workers'],
                                                                show_sample=False,
                                                                val_batch_size=Network_parameters['batch_size']['val'],
                                                                test_batch_size=Network_parameters['batch_size']['test'])
        dataloaders_dict = {'train': train_loader,'val': val_loader,'test': test_loader}


    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset)) 
    



    if Dataset=='LeavesTex' or Dataset=='DeepWeeds':
            pass
    
    else:
        image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=Network_parameters['batch_size'][x],
                                                        num_workers=Network_parameters['num_workers'],
                                                        pin_memory=Network_parameters['pin_memory'],
                                                        shuffle=False,
                                                        )
                                                        for x in ['train', 'val','test']}
                                                        


>>>>>>> adb24a22cd8f6ad31753938c1158b68666d89bbd
    return dataloaders_dict