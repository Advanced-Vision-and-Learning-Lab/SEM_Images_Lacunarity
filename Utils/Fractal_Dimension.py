import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

class FractalDimensionModule(nn.Module):
    def __init__(self):
        super(FractalDimensionModule, self).__init__()

    def boxcount_grayscale(self, Z, k):
        # Compute the local min and max pooling over the image
        S_min = -F.max_pool2d(-Z, kernel_size=k, stride=k)
        S_max = F.max_pool2d(Z, kernel_size=k, stride=k)
        return S_max - S_min
    
    def fractal_dimension_grayscale(self, Z):
        assert Z.dim() == 2, "Input must be a 2D tensor"
        
        # Minimal dimension of image
        p = min(Z.shape)
        
        # Greatest power of 2 less than or equal to p
        n = 2 ** int(torch.floor(torch.log2(torch.tensor(p))))
        
        # Build successive box sizes (from 2**n down to 2**1)
        sizes = 2 ** torch.arange(n.bit_length() - 1, 0, -1)
        
        # Box counting for each size
        i_difference = []
        for size in sizes:
            i_difference.append(self.boxcount_grayscale(Z.unsqueeze(0).unsqueeze(0), size.item()))
        
        # D_B (based on sum)
        d_b = torch.tensor([torch.sum(x).item() for x in i_difference])
        
        # D_M (based on mean)
        d_m = torch.tensor([torch.mean(x).item() for x in i_difference])
        
        # Log of sizes
        log_sizes = torch.log(sizes.float())
        
        # Log of d_b and d_m
        log_d_b = torch.log(d_b)
        log_d_m = torch.log(d_m)
        
        # Linear regression (least squares fitting using linalg.lstsq)
        A = torch.stack([log_sizes, torch.ones_like(log_sizes)], dim=1)
        
        # Coefficients for D_B
        coeffs_db, _ = torch.linalg.lstsq(A, log_d_b.unsqueeze(1)).solution.squeeze()
        
        # Coefficients for D_M
        coeffs_dm, _ = torch.linalg.lstsq(A, log_d_m.unsqueeze(1)).solution.squeeze()
        
        # Return the fractal dimensions (negative slope)
        return -coeffs_db, -coeffs_dm

    def forward(self, X):
        # If input is 2D, reshape to 4D (batch_size=1, channel=1)
        if X.dim() == 2:
            X = X.unsqueeze(0).unsqueeze(0)
        elif X.dim() == 3:
            X = X.unsqueeze(0)
        
        B, C, H, W = X.shape
        assert C == 1, "Input must have 1 channel"

        # Compute fractal dimensions for the entire image
        fractal_dimensions = self.fractal_dimension_grayscale(X.squeeze(0).squeeze(0))
        
        return fractal_dimensions
