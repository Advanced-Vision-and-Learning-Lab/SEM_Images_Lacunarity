import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_emd(hist1, hist2):
    hist1_np = hist1.cpu().numpy().astype(np.float32)
    hist2_np = hist2.cpu().numpy().astype(np.float32)
    
    # Create coordinate arrays for 2D histogram
    h, w = hist1_np.shape
    h1, w1 = hist2_np.shape
    #coords are dependent on quantization level
    coords = np.array([(i, j) for i in range(h) for j in range(w)], dtype=np.float32)
    
    # Flatten the 2D histograms
    hist1_flat = hist1_np.reshape(-1)
    hist2_flat = hist2_np.reshape(-1)
    
    # Calculate EMD
    emd_score, _, _ = cv2.EMD(
        np.column_stack((hist1_flat, coords)), 
        np.column_stack((hist2_flat, coords)), 
        cv2.DIST_L2
    )
    return emd_score

def calculate_emd_matrix(class_histograms):
    classes = list(class_histograms.keys())
    n_classes = len(classes)
    emd_matrix = np.zeros((n_classes, n_classes))
    
    for i, class1 in enumerate(classes):
        for j, class2 in enumerate(classes):
            if i != j:
                hist1 = torch.stack(class_histograms[class1]).mean(dim=0)
                hist2 = torch.stack(class_histograms[class2]).mean(dim=0)
                emd_matrix[i, j] = calculate_emd(hist1, hist2)
    return emd_matrix, classes

def visualize_emd_matrix(emd_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(emd_matrix, annot=True, fmt=".4f", cmap="YlGnBu", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Earth Mover's Distance (EMD) between Classes")
    plt.tight_layout()
    plt.show()
