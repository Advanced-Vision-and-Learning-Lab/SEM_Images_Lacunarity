# -*- coding: utf-8 -*-
"""
Main script for Lacunarity experiments - Modularized
"""
import argparse
import torch
import matplotlib.pyplot as plt
from Demo_Parameters import Parameters
from Prepare_Data import Prepare_DataLoaders
import os
import torch
import matplotlib.pyplot as plt
from View_Results import *
from Utils.Compute_EMD import *
from Utils.Local_aggregate import *
from Utils.Global_aggregate import *


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
plt.ioff()  # Turn off plotting

def main(Params):
    kernel = Params["kernel"]
    stride = Params["stride"]
    Dataset = Params['Dataset']
    texture_feature = Params['texture_feature']
    levels = Params["quant_levels"]
    agg_func = Params["agg_func"]

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloader
    dataset, loader = Prepare_DataLoaders(Network_parameters=Params)

    # Process based on the aggregation function type
    if agg_func == "global":
        process_global_aggregation(texture_feature, dataset, loader, device)
    elif agg_func == "local":
        process_local_aggregation(texture_feature, dataset, loader, kernel, stride, device, levels)


def parse_args():
    parser = argparse.ArgumentParser(description='Run Angular Losses and Baseline experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments(default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models',
                        help='Location to save models')
    parser.add_argument('--kernel', type=int, default=31,
                        help='Input kernel size')
    parser.add_argument('--stride', type=int, default=1,
                        help='Input stride size')
    parser.add_argument('--padding', type=int, default=0,
                        help='Input padding size')
    parser.add_argument('--quant_levels', type=int, default=5,
                        help='Input kernel size')
    parser.add_argument('--texture_feature', type=int, default=3,
                        help='texture_feature selection: 1:Fractal_Dimension, 2:Base_Lacunarity, 3:GAP')
    parser.add_argument('--agg_func', type=int, default=2,
                        help='agg func: 1:global, 2:local')
    parser.add_argument('--data_selection', type=int, default=1,
                        help='Dataset selection: 1:LungCells_DC')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    params = Parameters(args)
    main(params)
