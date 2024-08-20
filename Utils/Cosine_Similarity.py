import torch
import torch.nn.functional as F

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