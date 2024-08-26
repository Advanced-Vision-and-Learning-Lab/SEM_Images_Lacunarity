import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FractalDimensionModule(nn.Module):
    def __init__(self, window_size, stride=1):
        super(FractalDimensionModule, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.sizes = 2 ** torch.arange(int(np.log2(window_size)), 1, -1, dtype=torch.float32)
        self.log_sizes = torch.log(self.sizes)

    def boxcount_grayscale(self, Z):
        results = []
        for size in self.sizes:
            size = int(size.item())
            S_min = F.max_pool2d(-Z, size, stride=size) * -1
            S_max = F.max_pool2d(Z, size, stride=size)
            results.append((S_max - S_min).squeeze())
        return results

    def fractal_dimension_grayscale(self, differences):
        d_b = torch.tensor([x.sum().item() for x in differences])
        d_m = torch.tensor([x.mean().item() for x in differences])
        
        log_d_b = torch.log(d_b)
        log_d_m = torch.log(d_m)
        
        coeffs_db = torch.linalg.lstsq(self.log_sizes.unsqueeze(1), log_d_b.unsqueeze(1)).solution
        coeffs_dm = torch.linalg.lstsq(self.log_sizes.unsqueeze(1), log_d_m.unsqueeze(1)).solution
        
        return -coeffs_db.item(), -coeffs_dm.item()

    def forward(self, X):
        # Check if input is 4D (batch_size=1, channels=1, height, width)
        if len(X.shape) == 4:
            assert X.shape[0] == 1 and X.shape[1] == 1, "Input must have batch size 1 and 1 channel"
            X = X.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
        else:
            assert len(X.shape) == 2, "Input must be a 2D tensor or 4D tensor with batch size 1 and 1 channel"
        
        output_shape = (
            (X.shape[0] - self.window_size) // self.stride + 1,
            (X.shape[1] - self.window_size) // self.stride + 1
        )
        fractal_dimensions = torch.zeros(output_shape, dtype=torch.float32, device=X.device)
        
        unfold = F.unfold(X.unsqueeze(0).unsqueeze(0), 
                          kernel_size=self.window_size, 
                          stride=self.stride)
        windows = unfold.transpose(1, 2).reshape(-1, self.window_size, self.window_size)
        
        for i, window in enumerate(windows):
            differences = self.boxcount_grayscale(window.unsqueeze(0).unsqueeze(0))
            fd, _ = self.fractal_dimension_grayscale(differences)
            fractal_dimensions[i // output_shape[1], i % output_shape[1]] = fd
        
        # Return as a 4D tensor (batch_size=1, channels=1, height, width)
        return fractal_dimensions.unsqueeze(0).unsqueeze(0)