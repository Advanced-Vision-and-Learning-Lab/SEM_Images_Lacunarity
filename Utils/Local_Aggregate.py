from collections import defaultdict
from tqdm import tqdm
from Utils.Base_Lacunarity import Base_Lacunarity
from Utils.Fractal_Dimension import FractalDimensionModule
from scipy.stats import gaussian_kde
import torch.nn as nn
from Utils.Quantization import QCO_2d
from Utils.Cosine_Similarity import *
from View_Results import *
from Utils.Compute_EMD import *
import matplotlib.pyplot as plt

def process_local_aggregation(texture_feature, dataset, loader, kernel, stride, device, levels, Params):
    """Processes the local aggregation function."""
    texture_feature =  Params["texture_feature"]
    agg_func = Params["agg_func"]
    if texture_feature == "Base_Lacunarity":
        texture = Base_Lacunarity(kernel=(kernel, kernel), stride=(stride, stride)).to(device)
    elif texture_feature == "Fractal_Dimension":
        texture = FractalDimensionModule(window_size=kernel, stride=stride).to(device)
    elif texture_feature == "GAP":
        texture = nn.AvgPool2d((kernel, kernel), stride=(stride, stride), padding=(0, 0))

    qco_2d = QCO_2d(scale=1, level_num=levels).to(device)
    class_texture_maps = defaultdict(list)

    for images, labels in tqdm(loader, desc="Processing images"):
        images = images.to(device)
        features = texture(images) if texture_feature != "Fractal_Dimension" else texture(images)[0]

        for i in range(len(labels)):
            class_name = dataset.classes[labels[i]]
            class_texture_maps[class_name].append(features[i])

    aggregated_texture = aggregate_class_texture_maps(class_texture_maps)
    # display_aggregate_feature_maps(aggregated_texture, colormap="viridis")
    emd_difference  = visualize_aggregated_maps(aggregated_texture, qco_2d, texture_feature, agg_func)
    return emd_difference


def display_aggregate_feature_maps(aggregated_texture, colormap="coolwarm"):
    """
    Displays the aggregate feature maps for all classes side by side using a diverging color map.
    Feature maps are normalized between 0 and 1 using min-max normalization.

    Args:
        aggregated_texture (dict): A dictionary containing aggregated feature maps for each class.
        colormap (str): The name of the matplotlib colormap to use (default is 'coolwarm').
    """
    num_classes = len(aggregated_texture)
    
    # Create a figure with specific spacing
    fig = plt.figure(figsize=(15, 4))
    
    # Create a grid with specific width ratios and spacing
    gs = plt.GridSpec(1, num_classes + 1, figure=fig, 
                     width_ratios=[1]*num_classes + [0.05],
                     wspace=0.05)
    
    # Create axes for the images
    axes = []
    for i in range(num_classes):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
    
    # Create axis for colorbar
    cbar_ax = fig.add_subplot(gs[0, -1])
    
    # Plot images with equal aspect ratio
    for ax, (class_name, agg_map) in zip(axes, aggregated_texture.items()):
        transformed_map_np = agg_map.squeeze(0).cpu().numpy()
        
        # Apply min-max normalization
        p5 = np.percentile(transformed_map_np, 5)
        p95 = np.percentile(transformed_map_np, 95)
        normalized_map = np.clip((transformed_map_np - p5) / (p95 - p5), 0, 1)

        
        img = ax.imshow(normalized_map, 
                       cmap=colormap, 
                       vmin=0,  # Set fixed range for normalized values
                       vmax=1,
                       aspect='auto')  # Ensure images fill the space
        ax.set_title(class_name)
        ax.axis('off')
    
    # Add colorbar with specific formatting
    cbar = plt.colorbar(img, cax=cbar_ax)
    cbar_ax.set_ylabel('Normalized Feature Value', 
                      rotation=90,  # Vertical text
                      labelpad=25)  # Increase spacing from colorbar
    
    # Adjust layout to remove excess whitespace
    plt.subplots_adjust(wspace=0.3, right=0.92)
    
    # Display the plot
    plt.show()
    
    return fig, axes



def aggregate_class_texture_maps(class_texture_maps):
    """Aggregates the texture maps for each class."""
    aggregated_texture = {}
    for class_name, texture_maps in class_texture_maps.items():
        aggregated_texture[class_name] = aggregate_texture_maps(texture_maps, class_name)
        
    return aggregated_texture



def visualize_aggregated_maps(aggregated_texture, qco_2d, texture_feature, agg_func): 
    """Visualizes and processes the aggregated texture maps."""
    class_qco_outputs = {}
    class_sta_avgs = {}

    for class_name, agg_map in aggregated_texture.items():
        output, _, _ = qco_2d(agg_map)
        class_qco_outputs[class_name] = output
        class_sta_avgs[class_name] = output[:, 2]  # Extract statistical texture values

    # visualize_class_sta_distributions(class_sta_avgs)
    emd_matrix, class_names, difference = calculate_emd_matrix(class_qco_outputs)
    # visualize_emd_matrix(emd_matrix, class_names)


    ranking = rank_classes_by_emd(class_qco_outputs, reference_class='Untreated')

    toxicologist_ranking = ['Nickel Oxide (NiO)', 'Crystalline Silica (CS)']
    ranking_kappa = calculate_ranking_kappa(toxicologist_ranking, ranking)
    create_ranking_comparison_plot(toxicologist_ranking, ranking, ranking_kappa, texture_feature, agg_func)
    print("Ranking comparison plot saved as 'ranking_comparison.png")

    return difference
