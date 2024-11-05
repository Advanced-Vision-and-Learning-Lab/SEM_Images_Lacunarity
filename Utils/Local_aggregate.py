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
    visualize_aggregated_maps(aggregated_texture, qco_2d, texture_feature, agg_func)



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

    visualize_class_sta_distributions(class_sta_avgs)
    emd_matrix, class_names = calculate_emd_matrix(class_qco_outputs)
    visualize_emd_matrix(emd_matrix, class_names)


    ranking = rank_classes_by_emd(class_qco_outputs, reference_class='Untreated')

    toxicologist_ranking = ['Nickel Oxide (NiO)', 'Crystalline Silica (CS)']
    ranking_kappa = calculate_ranking_kappa(toxicologist_ranking, ranking)
    create_ranking_comparison_plot(toxicologist_ranking, ranking, ranking_kappa, texture_feature, agg_func)
    print("Ranking comparison plot saved as 'ranking_comparison.png")

