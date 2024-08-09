# -*- coding: utf-8 -*-
"""
Create datasets and dataloaders for models
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import pdb

## PyTorch dependencies
import torch
from torchvision import transforms
import torch.nn as nn

# Data augmentation and normalization for training
# Just normalization and resize for test
# Data transformations as described in:
# http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf

def get_transform(Network_parameters, input_size=224):
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    
    if Dataset == "LungCells_DC" or Dataset == 'LungCells_ME':
        data_transforms = {
            transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
}
    
    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset))
    
    return data_transforms
