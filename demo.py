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
from matplotlib.gridspec import GridSpec
from tabulate import tabulate
import torch.nn as nn
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance
import seaborn as sns

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#Turn off plotting
plt.ioff()

def main(Params):
   kernel = Params["kernel"]
   stride = Params["stride"]
   Dataset = Params['Dataset'] 
   texture_feature = Params['texture_feature']
   levels = Params["quant_levels"]
   class_qco_outputs = {}
   class_sta_avgs = {}
   agg_func = Params["agg_func"]
   
   # Detect if we have a GPU available
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   if agg_func == "global":
      if texture_feature == "Base_Lacunarity":
         texture = Base_Lacunarity().to(device)
      elif texture_feature == "Fractal_Dimension":
         texture = FractalDimensionModule().to(device)
      elif texture_feature == "GAP":
         texture = nn.AdaptiveAvgPool2d(1)

   elif agg_func == "local":
      if texture_feature == "Base_Lacunarity":
         texture = Base_Lacunarity(kernel=(kernel, kernel), stride=(stride, stride)).to(device)
      elif texture_feature == "Fractal_Dimension":
         texture = FractalDimensionModule(window_size=kernel, stride=stride).to(device)
      elif texture_feature == "GAP":
         texture = nn.AvgPool2d((kernel, kernel), stride=(stride, stride), padding=(0, 0))
   qco_2d = QCO_2d(scale=1, level_num=levels).to(device)

   # Initialize dataset and dataloader
   dataset, loader = Prepare_DataLoaders(Network_parameters=Params)

   # Initialize storage
   class_lacunarity_maps = defaultdict(list)
   class_global_features = defaultdict(list)

   # Process batches
   for images, labels in tqdm(loader, desc="Processing images"):
      images = images.to(device)
      features, _ = texture(images)

      for i in range(len(labels)):
         class_name = dataset.classes[labels[i]]
         if agg_func == "global":
               class_global_features[class_name].append(features.cpu().numpy().flatten())
         else:
               lacunarity_map = features[i]
               class_lacunarity_maps[class_name].append(lacunarity_map)

   if agg_func == "global":
      # Create KDE for GAP features
      class_kde = {}
      all_features = []
      for class_name, features_list in class_global_features.items():
         features = np.concatenate(features_list)
         all_features.extend(features)
         kde = gaussian_kde(features)
         class_kde[class_name] = kde

      # Visualize KDE histograms
      plt.figure(figsize=(10, 6))
      x_range = np.linspace(min(all_features), max(all_features), 1000)
      for class_name, kde in class_kde.items():
         plt.plot(x_range, kde(x_range), label=class_name)
      plt.title("KDE of Features by Class")
      plt.xlabel("Feature Value")
      plt.ylabel("Density")
      plt.legend()
      plt.savefig('gap_kde_histogram.png')
      plt.close()

      # Calculate EMD for GAP features
      emd_matrix = np.zeros((len(class_kde), len(class_kde)))
      class_names = list(class_kde.keys())
      for i, class1 in enumerate(class_names):
         for j, class2 in enumerate(class_names):
               if i != j:
                  # Generate samples from KDEs for EMD calculation
                  samples1 = class_kde[class1].resample(1000)[0]
                  samples2 = class_kde[class2].resample(1000)[0]
                  emd_matrix[i, j] = wasserstein_distance(samples1, samples2)

      # Visualize EMD matrix
      plt.figure(figsize=(10, 8))
      sns.heatmap(emd_matrix, annot=True, fmt=".4f", cmap="YlGnBu", 
                  xticklabels=class_names, yticklabels=class_names)
      plt.title("Earth Mover's Distance (EMD) between Classes")
      plt.tight_layout()
      plt.savefig('gap_emd_matrix.png')
      plt.close()

      print("GAP EMD Matrix:")
      print(pd.DataFrame(emd_matrix, index=class_names, columns=class_names))

   elif agg_func == "local":
      # Aggregate lacunarity maps for each class
      aggregated_lacunarity = {}
      for class_name, lacunarity_maps in class_lacunarity_maps.items():
         aggregated_lacunarity[class_name] = aggregate_lacunarity_maps(lacunarity_maps)

      # Visualize aggregated lacunarity maps
      visualize_representative_lacunarity(aggregated_lacunarity)
      
      # Pass aggregated lacunarity maps through QCO
      class_qco_outputs = {}
      for class_name, agg_map in aggregated_lacunarity.items():
         output, _, _ = qco_2d(agg_map)
         class_qco_outputs[class_name] = output
         # Extract only the statistical texture values (third column)
         class_sta_avgs[class_name] = output[:, 2]

      # Visualize class sta distributions
      visualize_class_sta_distributions(class_sta_avgs)

      # Calculate and visualize EMD matrix
      emd_matrix, class_names = calculate_emd_matrix(class_qco_outputs)
      print("EMD Matrix:")
      print(pd.DataFrame(emd_matrix, index=class_names, columns=class_names))
      visualize_emd_matrix(emd_matrix, class_names)

      emd_ranking = rank_classes_by_emd(class_qco_outputs, reference_class='Untreated')

      # Calculate Kappa between toxicologist and EMD-based rankings
      toxicologist_ranking = ['Nickel Oxide (NiO)', 'Silver Nanoparticles (Ag-NP)', 'Crystalline Silica (CS)']
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

      # Create and print EMD scores table
      emd_table = tabulate(emd_scores, headers=["Class", "EMD Score"], tablefmt="grid", floatfmt=".4f")
      print(emd_table)

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
    parser.add_argument('--texture_feature', type=int, default=1,
                        help='texture_feature selection: 1:Fractal_Dimension, 2:Base_Lacunarity, 3:GAP')
    parser.add_argument('--agg_func', type=int, default=1,
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