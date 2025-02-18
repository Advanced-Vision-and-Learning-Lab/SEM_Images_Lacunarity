# -*- coding: utf-8 -*-

## Python standard libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
## PyTorch dependencies
import torch
from scipy import stats
import pdb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

def create_ranking_comparison_plot(toxicologist_ranking, emd_ranking, kappa_score, texture_feature, agg_func):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data for the table
    table_data = []
    for i in range(max(len(toxicologist_ranking), len(emd_ranking))):
        tox_class = toxicologist_ranking[i] if i < len(toxicologist_ranking) else ""
        emd_class = emd_ranking[i] if i < len(emd_ranking) else ""
        table_data.append([i+1, tox_class, emd_class])

    # Add Kappa score row
    table_data.append(["", "", ""])
    table_data.append(["Kappa Score", f"{kappa_score:.4f}" if kappa_score is not None else "N/A", ""])

    # Create the table
    table = ax.table(cellText=table_data, 
                     colLabels=["Rank", "Toxicologist Ranking", "EMD-based Ranking"],
                     cellLoc='center', loc='center')

    # Modify table aesthetics
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Highlight header and Kappa score rows
    for (row, col), cell in table.get_celld().items():
        if row == 0 or row == len(table_data) - 1:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f0f0f0')

    # Add title
    plt.title(f"Ranking Comparison and Kappa Score\nPooling Layer: {texture_feature}, Aggregation: {agg_func}", fontsize=16, pad=20)

    # Save the figure with the custom file name based on pooling_layer and agg_func
    file_name = f'ranking_comparison_{texture_feature}__{agg_func}.png'
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()


def visualize_kde_histograms(class_kde, all_features):
    plt.figure(figsize=(10, 6))
    x_range = np.linspace(min(all_features), max(all_features), 1000)
    for class_name, kde in class_kde.items():
        plt.plot(x_range, kde(x_range), label=class_name)
    plt.title("KDE of Features by Class")
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

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
    plt.figure(figsize=(12, 10))  # Increased figure size for better readability
    
    # Create the heatmap
    ax = sns.heatmap(emd_matrix, annot=True, fmt=".4f", cmap="YlGnBu", 
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={'size': 20}) 
    
    # Increase font size for the annotations (numbers in cells)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    
    # Increase font size for colorbar labels
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    
    # Set title with larger font
    plt.title("Earth Mover's Distance (EMD) between Classes", fontsize=16)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def visualize_average_lacunarity(average_lacunarity_per_class):
    n_classes = len(average_lacunarity_per_class)
    fig, axes = plt.subplots(1, n_classes + 1, figsize=(20, 4), 
                             gridspec_kw={'width_ratios': [1]*n_classes + [0.05]})
    fig.suptitle('Aggregate Feature Maps by Class')

    # Determine global min and max for consistent color scaling
    all_values = torch.cat(list(average_lacunarity_per_class.values()))
    vmin, vmax = all_values.min().item(), all_values.max().item()

    for idx, (class_name, avg_lacunarity) in enumerate(average_lacunarity_per_class.items()):
        im = axes[idx].imshow(avg_lacunarity.squeeze(0).cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[idx].set_title(class_name)
        axes[idx].axis('off')

    # Add a colorbar to the right of the last image
    cbar = fig.colorbar(im, cax=axes[-1])
    cbar.set_label('Feature Value')

    plt.tight_layout()
    plt.show()


def visualize_representative_texture(representative_lacunarity):
    n_classes = len(representative_lacunarity)
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(4*n_classes, 4))
    gs = GridSpec(1, n_classes+1, width_ratios=[1]*n_classes + [0.05], figure=fig)
    
    # Determine global min and max for consistent color scaling
    all_values = torch.cat([map.flatten() for map in representative_lacunarity.values()])
    vmin, vmax = all_values.min().item(), all_values.max().item()

    # Create subplots
    for idx, (class_name, rep_lacunarity) in enumerate(representative_lacunarity.items()):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(rep_lacunarity.squeeze().cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(class_name)
        ax.axis('off')

    # Add colorbar
    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Feature Value')

    # Add main title
    fig.suptitle('Representative Feature Maps by Class', y=1.02)

    # Adjust layout
    plt.tight_layout()
    
    # Adjust the top of the subplots to make room for the suptitle
    plt.subplots_adjust(top=0.85)

    plt.show()



def visualize_class_sta_distributions(class_sta_avgs):
    data = []
    for class_name, sta_avg in class_sta_avgs.items():
        sta_np = sta_avg.cpu().numpy().flatten()
        data.extend([(class_name, val) for val in sta_np])
    
    df = pd.DataFrame(data, columns=['Class', 'Statistical Texture Value'])


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
