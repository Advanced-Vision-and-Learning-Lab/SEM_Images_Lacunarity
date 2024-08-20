import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


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