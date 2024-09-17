import torch
import torch.nn.functional as F
import pdb

def aggregate_lacunarity_maps(maps):
   # Normalize each map by subtracting the mean and dividing by the standard deviation
    normalized_maps = [(m - m.mean()) / (m.std() + 1e-8) for m in maps]

    # Flatten the normalized maps for similarity calculation
    flat_maps = torch.stack([m.view(-1) for m in normalized_maps])

    # Calculate the cosine similarity matrix for all pairs
    norm_flat_maps = F.normalize(flat_maps, p=2, dim=1)  # Normalize vectors to unit length (L2 norm)
    sim_matrix = torch.mm(norm_flat_maps, norm_flat_maps.t())  # Matrix multiplication for cosine similarity

    # Compute the weights as the mean of the similarity matrix along the rows
    weights = sim_matrix.mean(dim=1)

    # Weighted aggregation
    weighted_sum = sum(map * weight for map, weight in zip(normalized_maps, weights))
    
    return weighted_sum / weights.sum()

# def aggregate_lacunarity_maps(maps):
#     # Stack the maps into a single tensor
#     stacked_maps = torch.stack(maps)  # Shape: [C, H, W]
    
#     # Normalize each channel (map) independently
#     normalized_maps = (stacked_maps - stacked_maps.mean(dim=(1, 2), keepdim=True)) / (stacked_maps.std(dim=(1, 2), keepdim=True) + 1e-8)

#     # Apply global average pooling along H, W to get a pooled vector for each channel
#     pooled_maps = F.adaptive_avg_pool2d(stacked_maps, (1, 1))  # Shape: [C, 1, 1]
    
#     # Compute cosine similarity between the pooled vector and each pixel in the feature map
#     cos_sim = F.cosine_similarity(stacked_maps, pooled_maps, dim=0)  # Cosine similarity across channels, Shape: [H, W]


#     return cos_sim
