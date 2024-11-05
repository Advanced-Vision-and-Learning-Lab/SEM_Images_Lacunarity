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

def process_global_aggregation(texture_feature, dataset, loader, device, Params):
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
    """Visualizes and calculates EMD for the global aggregation method using KDE directly with OpenCV's cv2.EMD."""
    class_kde = {}
    all_features = []

    # Create KDEs for each class and store them in class_kde
    for class_name, features_list in class_global_features.items():
        features = np.concatenate(features_list)
        all_features.extend(features)
        kde = gaussian_kde(features)
        class_kde[class_name] = kde
    
    visualize_kde_histograms(class_kde, all_features)

    # Define common range for KDE evaluation
    min_val = min(all_features)
    max_val = max(all_features)
    bins = 50  # You can adjust the number of bins for KDE resolution
    bin_centers = np.linspace(min_val, max_val, bins)

    # Calculate EMD for GAP features using KDE directly
    emd_matrix = np.zeros((len(class_kde), len(class_kde)))
    class_names = list(class_kde.keys())
    
    for i, class1 in enumerate(class_names):
        for j, class2 in enumerate(class_names):
            if i != j:
                # Evaluate KDEs at the bin centers
                kde_values1 = class_kde[class1](bin_centers)
                kde_values2 = class_kde[class2](bin_centers)
                
                # Normalize KDE values to sum to 1 (to approximate probabilities)
                kde_values1 /= kde_values1.sum()
                kde_values2 /= kde_values2.sum()

                # Convert to signature format: each row [weight, position]
                signature1 = np.vstack([kde_values1, bin_centers]).T.astype(np.float32)
                signature2 = np.vstack([kde_values2, bin_centers]).T.astype(np.float32)

                # Calculate EMD using cv2.EMD
                emd, _, _ = cv2.EMD(signature1, signature2, cv2.DIST_L2)
                emd_matrix[i, j] = emd

    visualize_emd_matrix(emd_matrix, class_names)
    print("GAP EMD Matrix:")
    print(pd.DataFrame(emd_matrix, index=class_names, columns=class_names))

    # Return the EMD matrix and class names for further processing
    return emd_matrix, class_names
