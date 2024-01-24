# -*- coding: utf-8 -*-
"""
Functions to train/test model
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import copy

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
import pdb
import os
import torch.nn.functional as F
from Utils.LacunarityPoolingLayer import Pixel_Lacunarity, ScalePyramid_Lacunarity, BuildPyramid, DBC, GDCB, Base_Lacunarity


class Net(nn.Module):
    def __init__(self, num_ftrs, num_classes, Params, pooling_layer="lacunarity", agg_func="global"):

        super(Net, self).__init__()

        kernel = Params["kernel"]
        stride = Params["stride"]
        padding = Params["conv_padding"]
        scales = Params["scales"]
        num_levels = Params["num_levels"]
        sigma = Params["sigma"]
        min_size = Params["min_size"]
        bias = Params["bias"]

        self.agg_func = agg_func
        if agg_func == "global":
            if pooling_layer == "max":
                self.pooling_layer = nn.AdaptiveMaxPool2d(1)
            elif pooling_layer == "avg":
                self.pooling_layer = nn.AdaptiveAvgPool2d(1)
            elif pooling_layer == "Pixel_Lacunarity":
                self.pooling_layer = Pixel_Lacunarity(scales=scales, bias = bias)
        elif agg_func == "local":
            if pooling_layer == "max":
                self.pooling_layer = nn.MaxPool2d(kernel_size=(kernel, kernel), stride =(stride, stride), padding=(padding, padding))
            elif pooling_layer == "avg":                                                                                                                                                                                                                           
                self.pooling_layer = nn.AvgPool2d(kernel_size=(kernel, kernel), stride =(stride, stride), padding=(padding, padding))
            elif pooling_layer == "Base_Lacunarity":
                self.pooling_layer = Base_Lacunarity(scales=scales, kernel=(kernel, kernel), stride =(stride, stride), bias=bias)
            elif pooling_layer == "Pixel_Lacunarity":
                self.pooling_layer = Pixel_Lacunarity(scales=scales, kernel=(kernel, kernel), stride =(stride, stride), bias=bias)
            elif pooling_layer == "ScalePyramid_Lacunarity":
                self.pooling_layer = ScalePyramid_Lacunarity(num_levels=num_levels, sigma = sigma, min_size = min_size, kernel=(kernel, kernel), stride =(stride, stride))
            elif pooling_layer == "BuildPyramid":
                self.pooling_layer = BuildPyramid(num_levels=num_levels, kernel=(kernel, kernel), stride =(stride, stride))
            elif pooling_layer == "DBC":
                self.pooling_layer = DBC(r_values = scales, window_size = kernel)
            elif pooling_layer == "GDCB":
                self.pooling_layer = GDCB(3,5)

                """Scale_Lacunarity(kernel=(3,3), stride =(1,1))"""
                """ self.pooling_layer = Global_Lacunarity(scales=[i/10.0 for i in range(0, 20)], kernel=(4,4), stride =(1,1)) """
                """[i/10 for i in range(10, 21)]"""
                """[i/10 for i in range(10, 31)]"""


        self.conv1 = nn.Conv2d(3, out_channels=3, kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()
        self.fc = nn.Linear(num_ftrs, num_classes)
        



    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pooling_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x