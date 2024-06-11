<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Functions to train/test model
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
## PyTorch dependencies
import torch.nn as nn


class lacunarity_pooling(nn.Module):
    def __init__ (self, lacunarity_layer, Params):

        super(lacunarity_pooling, self).__init__()
        self.lacunarity_layer = lacunarity_layer
        self.model_name = Params["Model_name"]
        self.dataset = Params["Dataset"]
        self.num_classes = Params["num_classes"][self.dataset]
        self.feature_extraction = Params['feature_extraction']
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pooling_layer = self.lacunarity_layer

    def forward(self, x):
        x_pool = self.pooling_layer(x)
        x_avg = self.avgpool(x)
        pool_layer = x_pool * x_avg
    
=======
# -*- coding: utf-8 -*-
"""
Functions to train/test model
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
## PyTorch dependencies
import torch.nn as nn
import pdb


class lacunarity_pooling(nn.Module):
    def __init__ (self, lacunarity_layer, Params):

        super(lacunarity_pooling, self).__init__()
        self.lacunarity_layer = lacunarity_layer
        self.model_name = Params["Model_name"]
        self.dataset = Params["Dataset"]
        self.num_classes = Params["num_classes"][self.dataset]
        self.feature_extraction = Params['feature_extraction']
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pooling_layer = self.lacunarity_layer

    def forward(self, x):
        x_pool = self.pooling_layer(x)
        x_avg = self.avgpool(x)
        pool_layer = x_pool * x_avg
    
>>>>>>> adb24a22cd8f6ad31753938c1158b68666d89bbd
        return pool_layer