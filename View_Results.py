# -*- coding: utf-8 -*-

## Python standard libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
## PyTorch dependencies
import torch
from scipy import stats


def visualize_class_sta_distributions(class_sta_avgs):
    data = []
    for class_name, sta_avg in class_sta_avgs.items():
        sta_np = sta_avg.cpu().numpy().flatten()
        data.extend([(class_name, val) for val in sta_np])
    
    df = pd.DataFrame(data, columns=['Class', 'Statistical Texture Value'])

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Class', y='Statistical Texture Value', data=df)
    plt.title('Distribution of Average Statistical Texture Values by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Class', y='Statistical Texture Value', data=df)
    plt.title('Distribution of Average Statistical Texture Values by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    n_classes = len(class_sta_avgs)
    fig, axes = plt.subplots(1, n_classes + 1, figsize=(20 + 2, 4),
                             gridspec_kw={'width_ratios': [1] * n_classes + [0.1]})
    
    global_min = min(sta_avg.min().item() for sta_avg in class_sta_avgs.values())
    global_max = max(sta_avg.max().item() for sta_avg in class_sta_avgs.values())

    for i, (class_name, sta_avg) in enumerate(class_sta_avgs.items()):
        im = sns.heatmap(sta_avg.cpu().numpy(), ax=axes[i], cmap='viridis',
                         vmin=global_min, vmax=global_max, cbar=False)
        axes[i].set_title(class_name)
        axes[i].axis('off')
    
    fig.colorbar(im.collections[0], cax=axes[-1], orientation='vertical')
    axes[-1].set_ylabel('Statistical Texture Value')
    
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 7))
    for class_name, sta_avg in class_sta_avgs.items():
        data = sta_avg.cpu().numpy().flatten()
        plt.hist(data, bins=10, alpha=0.3, density=True, label=f'{class_name} (Histogram)')
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        plt.plot(x_range, kde(x_range), label=f'{class_name} (KDE)')

    plt.title('Distribution of Average Statistical Texture Values by Class (Histogram and KDE)')
    plt.xlabel('Statistical Texture Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_emd_matrix(emd_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(emd_matrix, annot=True, fmt=".4f", cmap="YlGnBu", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Earth Mover's Distance (EMD) between Classes")
    plt.tight_layout()
    plt.show()


def visualize_average_lacunarity(average_lacunarity_per_class):
    n_classes = len(average_lacunarity_per_class)
    fig, axes = plt.subplots(1, n_classes + 1, figsize=(20, 4), 
                             gridspec_kw={'width_ratios': [1]*n_classes + [0.05]})
    fig.suptitle('Average Lacunarity Feature Maps by Class')

    # Determine global min and max for consistent color scaling
    all_values = torch.cat(list(average_lacunarity_per_class.values()))
    vmin, vmax = all_values.min().item(), all_values.max().item()

    for idx, (class_name, avg_lacunarity) in enumerate(average_lacunarity_per_class.items()):
        im = axes[idx].imshow(avg_lacunarity.squeeze(0).cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[idx].set_title(class_name)
        axes[idx].axis('off')

    # Add a colorbar to the right of the last image
    cbar = fig.colorbar(im, cax=axes[-1])
    cbar.set_label('Lacunarity Value')

    plt.tight_layout()
    plt.show()



def visualize_representative_lacunarity(representative_lacunarity):
    n_classes = len(representative_lacunarity)
    fig, axes = plt.subplots(1, n_classes + 1, figsize=(20, 4), 
                             gridspec_kw={'width_ratios': [1]*n_classes + [0.05]})
    fig.suptitle('Representative Lacunarity Feature Maps by Class')

    # Determine global min and max for consistent color scaling
    all_values = torch.cat([map.flatten() for map in representative_lacunarity.values()])
    vmin, vmax = all_values.min().item(), all_values.max().item()

    for idx, (class_name, rep_lacunarity) in enumerate(representative_lacunarity.items()):
        im = axes[idx].imshow(rep_lacunarity.squeeze().cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[idx].set_title(class_name)
        axes[idx].axis('off')

    # Add a colorbar to the right of the last image
    cbar = fig.colorbar(im, cax=axes[-1])
    cbar.set_label('Lacunarity Value')

    plt.tight_layout()
    plt.show()

