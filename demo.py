# -*- coding: utf-8 -*-
"""
Main script for Lacunarity experiments
"""
import argparse
import torch
import matplotlib.pyplot as plt
from Demo_Parameters import Parameters
from Prepare_Data import Prepare_DataLoaders
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pdb
from Utils.Base_Lacunarity import Base_Lacunarity
from Utils.Fractal_Dimension import FractalDimensionModule
from View_Results import *
from Prepare_Data import Prepare_DataLoaders
from Utils.Quantization import QCO_2d
from Utils.Cosine_Similarity import aggregate_lacunarity_maps
from Utils.Compute_EMD import *
import pandas as pd
from Utils.Compute_EMD import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'



def process_lacunarity_data(lacunarity_data):
    average_lacunarity = {
        class_name: torch.mean(torch.stack(maps), dim=0)
        for class_name, maps in lacunarity_data.items()
    }
    return average_lacunarity

def main(Params):
    kernel = Params["kernel"]
    stride = Params["stride"]
    Dataset = Params['Dataset'] 
    texture_feature = Params['texture_feature']
    levels = Params["quant_levels"]
    agg_func = Params["agg_func"]
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize texture feature extractor
    if agg_func == "global":
        if texture_feature == "Base_Lacunarity":
            texture = Base_Lacunarity().to(device)
        elif texture_feature == "Fractal_Dimension":
            texture = FractalDimensionModule().to(device)
        elif texture_feature == "GAP":
            texture = torch.nn.AdaptiveAvgPool2d(1)
    elif agg_func == "local":
        if texture_feature == "Base_Lacunarity":
            texture = Base_Lacunarity(kernel=(kernel, kernel), stride=(stride, stride)).to(device)
        elif texture_feature == "Fractal_Dimension":
            texture = FractalDimensionModule(window_size=kernel, stride=stride).to(device)
        elif texture_feature == "GAP":
            texture = torch.nn.AvgPool2d((kernel, kernel), stride=(stride, stride), padding=(0, 0))

    qco_2d = QCO_2d(scale=1, level_num=levels).to(device)

    # Initialize dataset and dataloader
    dataset, loader = Prepare_DataLoaders(Network_parameters=Params)

    # Initialize storage
    class_lacunarity_maps = defaultdict(list)
    class_global_features = defaultdict(list)

    # Process batches
    for images, labels in tqdm(loader, desc="Processing images"):
        images = images.to(device)
        if texture_feature == "Fractal_Dimension":
            features, _ = texture(images)
        else:
            features = texture(images)

        for i in range(len(labels)):
            class_name = dataset.classes[labels[i]]
            if agg_func == "global":
                class_global_features[class_name].append(features[i].cpu().numpy().flatten())
            else:
                class_lacunarity_maps[class_name].append(features[i])

    if agg_func == "global":
        # Process global features
        class_histograms = {class_name: np.concatenate(features_list) 
                            for class_name, features_list in class_global_features.items()}
        
        visualize_class_sta_distributions(pd.DataFrame({
            'Class': np.repeat(list(class_histograms.keys()), [len(f) for f in class_histograms.values()]),
            'Statistical Texture Value': np.concatenate(list(class_histograms.values()))
        }))

        # Calculate EMD matrix for global features
        emd_matrix, class_names = calculate_emd_matrix(class_histograms)
        visualize_emd_matrix(emd_matrix, class_names)

        print("Global Features EMD Matrix:")
        print(pd.DataFrame(emd_matrix, index=class_names, columns=class_names))

    elif agg_func == "local":
        # Process local features
        average_lacunarity = process_lacunarity_data(class_lacunarity_maps)
        visualize_average_lacunarity(average_lacunarity)
        
        # Compute QCO outputs and STA averages
        class_qco_outputs = {}
        class_sta_avgs = {}
        for class_name, avg_map in average_lacunarity.items():
            output, _, _ = qco_2d(avg_map)
            class_qco_outputs[class_name] = output
            class_sta_avgs[class_name] = output[:, 2]

        # Visualize class STA distributions
        visualize_class_sta_distributions(class_sta_avgs)

        # Compute and visualize EMD matrix
        emd_matrix, class_names = calculate_emd_matrix(class_qco_outputs)
        visualize_emd_matrix(emd_matrix, class_names)

        print("Local Features EMD Matrix:")
        print(pd.DataFrame(emd_matrix, index=class_names, columns=class_names))

        # Rank classes and compare with toxicologist ranking
        emd_ranking = rank_classes_by_emd(class_qco_outputs, reference_class='Untreated')
        toxicologist_ranking = ['Nickel Oxide (NiO)', 'Crystalline Silica (CS)']
        ranking_kappa = calculate_ranking_kappa(toxicologist_ranking, emd_ranking)

        # Create and save the ranking comparison plot
        create_ranking_comparison_plot(toxicologist_ranking, emd_ranking, ranking_kappa)
        print("Ranking comparison plot saved as 'ranking_comparison.png'")

        # Calculate and print EMD scores
        reference_qco = class_qco_outputs['Untreated']
        print("\nEMD Scores (relative to Untreated):")
        emd_scores = []
        for class_name in emd_ranking:
            if class_name != 'Untreated':
                emd_score = calculate_emd(class_qco_outputs[class_name], reference_qco)
                emd_scores.append([class_name, emd_score])

        # Print EMD scores table
        print(pd.DataFrame(emd_scores, columns=["Class", "EMD Score"]).to_string(index=False))

def parse_args():
    parser = argparse.ArgumentParser(description='Run Angular Losses and Baseline experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments(default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models',
                        help='Location to save models')
    parser.add_argument('--kernel', type=int, default=21,
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    params = Parameters(args)
    main(params)