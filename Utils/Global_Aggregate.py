from collections import defaultdict
from tqdm import tqdm
from Utils.Base_Lacunarity import Base_Lacunarity
from Utils.Fractal_Dimension import FractalDimensionModule
from scipy.stats import gaussian_kde
import torch.nn as nn
from scipy.stats import wasserstein_distance
import numpy as np
from View_Results import *
import pandas as pd
from Utils.Compute_EMD import *
import pdb
from tqdm import tqdm

def process_global_aggregation(texture_feature, dataset, loader, device, levels, Params):
    """Processes the global aggregation function and ranks classes based on EMD from the untreated class."""
    texture_feature = Params["texture_feature"]
    agg_func = Params["agg_func"]

    # Initialize texture feature model
    if texture_feature == "Base_Lacunarity":
        texture = Base_Lacunarity().to(device)
    elif texture_feature == "Fractal_Dimension":
        texture = FractalDimensionModule().to(device)
    elif texture_feature == "GAP":
        texture = nn.AdaptiveAvgPool2d(1)

    class_global_features = defaultdict(list)

    # Process images and extract features
    for images, labels in tqdm(loader, desc="Processing images"):
        images = images.to(device)
        features = texture(images) if texture_feature != "Fractal_Dimension" else texture(images)[0]

        for i in range(len(labels)):
            class_name = dataset.classes[labels[i]]
            class_global_features[class_name].append(features.cpu().numpy().flatten())

    # Visualize KDE and calculate EMD matrix
    emd_matrix, class_names = visualize_and_calculate_emd_for_global(class_global_features)

    # Get index of the untreated class
    untreated_class = "Untreated"
    untreated_index = class_names.index(untreated_class)

    # Calculate EMD ranking relative to untreated class
    emd_ranking = [
        class_name for class_name in sorted(
            class_names,
            key=lambda x: emd_matrix[class_names.index(x), untreated_index],
            reverse=True
        ) if class_name != untreated_class
    ]

    # Define the toxicologist ranking
    toxicologist_ranking = ['Nickel Oxide (NiO)', 'Crystalline Silica (CS)']
    ranking_kappa = calculate_ranking_kappa(toxicologist_ranking, emd_ranking)

    # Create the ranking comparison plot
    create_ranking_comparison_plot(toxicologist_ranking, emd_ranking, ranking_kappa, texture_feature, agg_func)
    print("Ranking comparison plot saved as 'ranking_comparison.png'")




def visualize_and_calculate_emd_for_global(class_global_features):
    """Calculate EMD between each pair of classes in class_global_features."""
    class_names = list(class_global_features.keys())
    num_classes = len(class_names)
    emd_matrix = np.zeros((num_classes, num_classes))  # EMD matrix

    for i, class_i in enumerate(class_names):
        for j, class_j in enumerate(class_names):
            if i >= j:  # No need to calculate for symmetric pairs
                continue
            
            # Flatten features for comparison
            features_i = np.concatenate(class_global_features[class_i], axis=0)
            features_j = np.concatenate(class_global_features[class_j], axis=0)
            
            # Calculate 1D EMD using Wasserstein Distance
            emd = wasserstein_distance(features_i, features_j)
            emd_matrix[i, j] = emd
            emd_matrix[j, i] = emd  # Symmetric

    visualize_emd_matrix(emd_matrix, class_names)

    return emd_matrix, class_names


