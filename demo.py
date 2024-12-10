# -*- coding: utf-8 -*-
"""
Main script for Lacunarity experiments - Modularized with Automated Iteration
"""
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from Demo_Parameters import Parameters
from Prepare_Data import Prepare_DataLoaders
import os
from View_Results import *
from Utils.Compute_EMD import *
from Utils.Local_aggregate import *
from Utils.Global_aggregate import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
plt.ioff()  # Turn off plotting

def main(Params):
    # Extract lists of kernel sizes and quantization levels from Params
    kernel_sizes = Params["kernel_sizes"]
    quant_levels = Params["quant_levels"]
    stride = Params["stride"]
    Dataset = Params['Dataset']
    texture_feature = Params['texture_feature']
    agg_func = Params["agg_func"]

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloader
    dataset, loader = Prepare_DataLoaders(Network_parameters=Params)

    # Loop through each combination of kernel size and quantization level
    results = {}  # Dictionary to store results for each combination
    for kernel in kernel_sizes:
        for level in quant_levels:
            Params["kernel"] = kernel
            Params["quant_levels"] = level

            # Process based on the aggregation function type
            if agg_func == "global":
                result = process_global_aggregation(texture_feature, dataset, loader, device, level, Params)
            elif agg_func == "local":
                print(f"Processing with kernel size: {kernel}, quantization level: {level}")
                result = process_local_aggregation(texture_feature, dataset, loader, kernel, stride, device, level, Params)

            # Store the result for this combination
            results[(kernel, level)] = result

    return results  # Return all results

def parse_args():
    parser = argparse.ArgumentParser(description='Run Angular Losses and Baseline experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments(default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models',
                        help='Location to save models')
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[7],
                        help='List of kernel sizes to iterate over')
    parser.add_argument('--stride', type=int, default=1,
                        help='Input stride size')
    parser.add_argument('--padding', type=int, default=0,
                        help='Input padding size')
    parser.add_argument('--quant_levels', nargs='+', type=int, default=[6],
                        help='List of quantization levels to iterate over')
    parser.add_argument('--texture_feature', type=int, default=2,
                        help='Texture feature selection: 1:Fractal_Dimension, 2:Base_Lacunarity, 3:GAP')
    parser.add_argument('--agg_func', type=int, default=1,
                        help='Aggregation function: 1:global, 2:local')
    parser.add_argument('--data_selection', type=int, default=1,
                        help='Dataset selection: 1:LungCells_DC')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Enables CUDA training')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Prepare parameters dictionary with lists for kernel_sizes and quant_levels
    params = Parameters(args)
    params["kernel_sizes"] = args.kernel_sizes
    params["quant_levels"] = args.quant_levels
    
    # Run main with automated iteration
    results = main(params)


    if args.agg_func == "local":
        def create_heatmap():
            # Extract unique kernel sizes and quantization levels
            kernel_sizes = sorted(set(k[0] for k in results.keys()))
            quant_levels = sorted(set(k[1] for k in results.keys()))
            
            # Create the matrix for the heatmap
            matrix = np.zeros((len(kernel_sizes), len(quant_levels)))
            for i, k in enumerate(kernel_sizes):
                for j, q in enumerate(quant_levels):
                    matrix[i, j] = results.get((k, q), np.nan)
            
            # Set up the plot with a larger figure size
            plt.figure(figsize=(15, 12))
            
            # Create custom colormap with vibrant colors
            cmap = sns.diverging_palette(
                h_neg=10,    # Red hue
                h_pos=130,   # Green hue
                s=100,       # Maximum saturation
                l=50,        # Medium lightness
                sep=1,       # Maximum separation
                n=256        # Number of colors
            )
            
            # Create the heatmap
            ax = sns.heatmap(
                matrix,
                cmap=cmap,
                center=0,
                vmin=-1,
                vmax=1,
                annot=True,
                fmt='.2f',
                square=True,
                xticklabels=quant_levels,
                yticklabels=kernel_sizes,
                cbar_kws={'label': 'EMD Difference'},
                annot_kws={'size': 20},  # Increase annotation font size
            )

            colorbar = ax.collections[0].colorbar
            colorbar.ax.tick_params(labelsize=12)  # Set font size for color bar tick labels
            colorbar.ax.set_ylabel('EMD Difference', fontsize=18)  # Set font size for the label

            # Increase tick label font sizes
            plt.xticks(fontsize=18)  # Increase x-axis tick labels font size
            plt.yticks(fontsize=18)  # Increase y-axis tick labels font size

            # If you also want to increase the axis labels font size:
            plt.xlabel('Quantization Levels (Q)', labelpad=10, fontsize=24)
            plt.ylabel('Kernel Size (K)', labelpad=10, fontsize=24)
            plt.title('EMD Difference Heatmap: Positive (Green) vs Negative (Red) Rankings', pad=20, fontsize=24)

            # Adjust layout
            plt.tight_layout()
            
            return plt

        # Create and display the plot
        plt.style.use('seaborn')
        fig = create_heatmap()
        plt.show()