import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class QCO_2d(nn.Module):
    def __init__(self, scale, level_num):
        super(QCO_2d, self).__init__()
        self.level_num = level_num
        self.scale = scale

    def forward(self, x):
        N, H, W = x.shape
        min, max = x.min(), x.max()
        q_levels = torch.linspace(min, max, self.level_num).to(x.device)
        q_levels = q_levels.view(1, 1, -1)
        x_reshaped = x.view(N, 1, H*W)
        sigma = 1 / (self.level_num / 2)
        quant = torch.exp(-(x_reshaped.unsqueeze(-1) - q_levels)**2 / (sigma**2))
        
        quant = quant.view(N, H, W, self.level_num).permute(0, 3, 1, 2)
        quant = F.pad(quant, (0, 1, 0, 1), mode='constant', value=0.)
        
        quant_left = quant[:, :, :H, :W].unsqueeze(2)
        quant_right = quant[:, :, 1:, 1:].unsqueeze(1)
        
        co_occurrence = quant_left * quant_right
        sta = co_occurrence.sum(dim=(-1, -2))
        sta = sta / sta.sum(dim=(1, 2))
        q_levels_h = q_levels.expand(N, self.level_num, self.level_num)
        q_levels_w = q_levels_h.permute(0, 2, 1)
        
        output = torch.stack([q_levels_h, q_levels_w, sta], dim=1)
        output = output.flatten(2).squeeze(0)
        
        return output, quant.squeeze(0), co_occurrence.squeeze(0)