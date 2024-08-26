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
from collections import defaultdict
from tqdm import tqdm
import pdb
from Utils.Base_Lacunarity import Base_Lacunarity
from Utils.Fractal_Dimension import FractalDimensionModule
from View_Results import *
from Prepare_Data import Prepare_DataLoaders
from Utils.Quantization import QCO_2d
from Utils.Cosine_Similarity import aggregate_lacunarity_maps
from Utils.Compute_EMD import calculate_emd_matrix

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#Turn off plotting
plt.ioff()


def main(Params):
  
   # Name of dataset
   kernel = Params["kernel"]
   stride = Params["stride"]
   Dataset = Params['Dataset'] 
   texture_feature = Params['texture_feature']
   
   # Detect if we have a GPU available
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   if texture_feature == "Base_Lacunarity":
      texture = Base_Lacunarity(kernel=(kernel, kernel), stride=(stride, stride)).to(device)
   elif texture_feature == "Fractal_Dimension":
      texture = FractalDimensionModule(window_size=kernel, stride=stride).to(device)
   qco_2d = QCO_2d(scale=1, level_num=5).to(device)

   # Initialize dataset and dataloader
   dataset, loader = Prepare_DataLoaders(Network_parameters=Params)

   # Initialize storage
   class_lacunarity_maps = defaultdict(list)

   # Process batches
   for images, labels in tqdm(loader, desc="Processing images"):
      images = images.to(device)
      base_values = texture(images)

      for i in range(len(labels)):
         class_name = dataset.classes[labels[i]]
         lacunarity_map = base_values[i]
         class_lacunarity_maps[class_name].append(lacunarity_map)

   # Aggregate lacunarity maps for each class
   aggregated_lacunarity = {}
   for class_name, lacunarity_maps in class_lacunarity_maps.items():
      aggregated_lacunarity[class_name] = aggregate_lacunarity_maps(lacunarity_maps)

   # # Visualize aggregated lacunarity maps
   # visualize_representative_lacunarity(aggregated_lacunarity)
   
   # Pass aggregated lacunarity maps through QCO
   class_qco_outputs = {}
   for class_name, agg_map in aggregated_lacunarity.items():
      output, _, _ = qco_2d(agg_map)
      class_qco_outputs[class_name] = output

   # Calculate and visualize EMD matrix
   emd_matrix, class_names = calculate_emd_matrix(class_qco_outputs)
   print("EMD Matrix:")
   print(pd.DataFrame(emd_matrix, index=class_names, columns=class_names))
   visualize_emd_matrix(emd_matrix, class_names)
     


def parse_args():
   parser = argparse.ArgumentParser(description='Run Angular Losses and Baseline experiments for dataset')
   parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                       help='Save results of experiments(default: True)')
   parser.add_argument('--folder', type=str, default='Saved_Models',
                       help='Location to save models')
   parser.add_argument('--kernel', type=int, default=63,
                       help='Input kernel size')
   parser.add_argument('--stride', type=int, default=16,
                       help='Input stride size')
   parser.add_argument('--padding', type=int, default=0,
                       help='Input padding size')
   parser.add_argument('--texture_feature', type=int, default=1,
                       help='texture_feature selection: 1:Fractal_Dimension, 2:Base_Lacunarity')
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

