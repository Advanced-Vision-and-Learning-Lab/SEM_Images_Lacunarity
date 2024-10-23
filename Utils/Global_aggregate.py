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

def process_global_aggregation(texture_feature, dataset, loader, device):
    """Processes the global aggregation function."""
    if texture_feature == "Base_Lacunarity":
        texture = Base_Lacunarity().to(device)
    elif texture_feature == "Fractal_Dimension":
        texture = FractalDimensionModule().to(device)
    elif texture_feature == "GAP":
        texture = nn.AdaptiveAvgPool2d(1)

    class_global_features = defaultdict(list)

    for images, labels in tqdm(loader, desc="Processing images"):
        images = images.to(device)
        features = texture(images) if texture_feature != "Fractal_Dimension" else texture(images)[0]

        for i in range(len(labels)):
            class_name = dataset.classes[labels[i]]
            class_global_features[class_name].append(features.cpu().numpy().flatten())

    visualize_and_calculate_emd_for_global(class_global_features)



def visualize_and_calculate_emd_for_global(class_global_features):
    """Visualizes and calculates EMD for the global aggregation method."""
    class_kde = {}
    all_features = []

    for class_name, features_list in class_global_features.items():
        features = np.concatenate(features_list)
        all_features.extend(features)
        kde = gaussian_kde(features)
        class_kde[class_name] = kde
    
    visualize_kde_histograms(class_kde, all_features)

    # Calculate EMD for GAP features
    emd_matrix = np.zeros((len(class_kde), len(class_kde)))
    class_names = list(class_kde.keys())
    for i, class1 in enumerate(class_names):
        for j, class2 in enumerate(class_names):
            if i != j:
                samples1 = class_kde[class1].resample(1000)[0]
                samples2 = class_kde[class2].resample(1000)[0]
                emd_matrix[i, j] = wasserstein_distance(samples1, samples2)
    
    visualize_emd_matrix(emd_matrix, class_names)
    print("GAP EMD Matrix:")
    print(pd.DataFrame(emd_matrix, index=class_names, columns=class_names))