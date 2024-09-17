import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
import pdb

# Helper functions
def calculate_emd(qco_output1, qco_output2):
    # Ensure the count is the first column, then the bin centers
    qco_output1 = qco_output1.transpose(0, 1) 
    qco_output2 = qco_output2.transpose(0, 1)
    qco_output1_swapped = torch.cat((qco_output1[:, 2:], qco_output1[:, :2]), dim=1)
    qco_output2_swapped = torch.cat((qco_output2[:, 2:], qco_output2[:, :2]), dim=1)
    
    qco_output1_np = qco_output1_swapped.cpu().numpy().astype(np.float32)
    qco_output2_np = qco_output2_swapped.cpu().numpy().astype(np.float32)
    
    emd_score, _, _ = cv2.EMD(qco_output1_np, qco_output2_np, cv2.DIST_L2)
    
    return emd_score

def rank_classes_by_emd(class_qco_outputs, reference_class='Untreated'):
    emd_scores = {}
    reference_qco = class_qco_outputs[reference_class]
    
    for class_name, qco_output in class_qco_outputs.items():
        if class_name != reference_class:
            emd_scores[class_name] = calculate_emd(qco_output, reference_qco)
    
    # Sort classes by EMD score (higher score means more dissimilar)
    ranked_classes = sorted(emd_scores.items(), key=lambda x: x[1], reverse=True)
    return [class_name for class_name, _ in ranked_classes]

def calculate_emd_matrix(class_histograms):
    classes = list(class_histograms.keys())
    n_classes = len(classes)
    emd_matrix = np.zeros((n_classes, n_classes))
    
    for i, class1 in enumerate(classes):
        for j, class2 in enumerate(classes):
            if i != j:
                hist1 = class_histograms[class1]
                hist2 = class_histograms[class2]
                emd_matrix[i, j] = calculate_emd(hist1, hist2)
    return emd_matrix, classes


def calculate_ranking_kappa(toxicologist_ranking, emd_ranking):
    # Convert rankings to numeric values
    tox_rank = {class_name: rank for rank, class_name in enumerate(toxicologist_ranking)}
    emd_rank = {class_name: rank for rank, class_name in enumerate(emd_ranking)}

    # Ensure both rankings have the same classes
    common_classes = set(tox_rank.keys()) & set(emd_rank.keys())
    
    tox_values = [tox_rank[cls] for cls in common_classes]
    emd_values = [emd_rank[cls] for cls in common_classes]
    
    # Print diagnostic information
    print("Toxicologist values:", tox_values)
    print("EMD values:", emd_values)
    
    # Check if all values are the same
    if len(set(tox_values)) == 1 and len(set(emd_values)) == 1:
        return 1.0 if tox_values == emd_values else 0.0
    
    # Calculate Kappa
    try:
        kappa = cohen_kappa_score(tox_values, emd_values)
        return kappa
    except ValueError as e:
        print(f"Error calculating Kappa: {e}")
        return None