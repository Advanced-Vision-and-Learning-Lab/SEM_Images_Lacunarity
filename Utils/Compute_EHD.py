# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:52:40 2020
Function to generate EHD feature maps
@author: jpeeples
"""
import numpy as np
from scipy import signal,ndimage
import torch.nn.functional as F
import torch
from skimage import data, img_as_float
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import math

def Generate_masks(mask_size=3,angle_res=45,normalize=False,rotate=False):
    
    #Make sure masks are appropiate size. Should not be less than 3x3 and needs
    #to be odd size
    if type(mask_size) is list:
        mask_size = mask_size[0]
    if mask_size < 3:
        mask_size = 3
    elif ((mask_size % 2) == 0):
        mask_size = mask_size + 1
    else:
        pass
    
    #EHD uses the Sobel kernel operator, but other edge operators could be used
    if mask_size == 3:
        if rotate:
            Gy = np.outer(np.array([1,2,1]).T,np.array([1,0,-1]))
        else:
            Gy = np.outer(np.array([1,0,-1]).T,np.array([1,2,1]))
    else:
        if rotate:
            Gy = np.outer(np.array([1,2,1]).T,np.array([1,0,-1]))
        else:
            Gy = np.outer(np.array([1,0,-1]).T,np.array([1,2,1]))
        dim = np.arange(5,mask_size+1,2)
        expand_mask =  np.outer(np.array([1,2,1]).T,np.array([1,2,1]))
        for size in dim:
            Gy = signal.convolve2d(expand_mask,Gy)
    
    #Generate horizontal masks
    angles = np.arange(0,360,angle_res)
    masks = np.zeros((len(angles),mask_size,mask_size))
    
    #TBD: May need to improve for larger masks sizes 
    for rot_angle in range(0,len(angles)):
        masks[rot_angle,:,:] = ndimage.rotate(Gy,angles[rot_angle],reshape=False,
                                              mode='nearest')
        
    #Normalize masks if desired
    if normalize:
        if mask_size == 3:
            masks = (1/8) * masks
        else:
            masks = (1/8) * (1/16)**len(dim) * masks 
    return masks


def Get_EHD(X,args,device=None):
    
    #Generate masks based on parameters
    masks = Generate_masks(mask_size=args.kernel_size,
                           angle_res=args.angle_res,
                           normalize=args.normalize_kernel)
  
    #Convolve input with filters, expand masks to match input channels
    #TBD works for grayscale images (single channel input)
    #Check for multi-image input
    in_channels = X.shape[1]
    masks = torch.tensor(masks)
    masks = masks.unsqueeze(1)
    
    #Replicate masks along channel dimension
    masks = masks.repeat(1,in_channels,1,1)
    
    if device is not None:
        masks = masks.to(device)
  
    edge_responses = F.conv2d(X,masks,stride=args.stride_edge,
                              dilation=args.dilation)
    
    #Find max response
    [value,index] = torch.max(edge_responses,dim=1)
    
    #Set edge responses to "no edge" if not larger than threshold
    index[value<args.threshold] = masks.shape[0] 
    
    feat_vect = []
    window_scale = np.prod(np.asarray(args.window_size))
    
    #Generate counts that correspond to max edge response
    for edge in range(0,masks.shape[0]+1):
        # #Sum count
        if args.normalize_count:
           #Average count
            feat_vect.append((F.avg_pool2d((index==edge).unsqueeze(1).float(),
                              args.window_size,stride=args.stride_count,
                              count_include_pad=False).squeeze(1)))
        else:
            feat_vect.append(window_scale*F.avg_pool2d((index==edge).unsqueeze(1).float(),
                              args.window_size,stride=args.stride_count,
                              count_include_pad=False).squeeze(1))
        
    
    #Return vector
    feat_vect = torch.stack(feat_vect,dim=1)
        
    return feat_vect
#increase window size to figure out
def parse_args():
    parser = argparse.ArgumentParser(description='Run EHD example')
    parser.add_argument('--kernel_size', type=list, default=[21, 21],
                        help='Convolution kernel size for edge responses')
    parser.add_argument('--window_size', type=list, default=[7,7],
                        help='Binning count window size')
    parser.add_argument('--angle_res', type=int, default=45,
                        help='Angle resolution for masks rotations')
    parser.add_argument('--normalize_count', type=bool, default=False,
                        help='Flag to use sum (unnormalized count) or average pooling (normalized count)')
    parser.add_argument('--normalize_kernel', type=bool, default=False,
                        help='Flag to use Sobel kernel or normalized Sobel kernel')
    parser.add_argument('--threshold', type=float, default=1/8,
                        help='Threshold for no edge orientation (keep "strong" edges), good rule of thumb is 1/number of edge orientations')
    parser.add_argument('--stride_count', type=int, default=1,
                        help='Stride for count')
    parser.add_argument('--stride_edge', type=int, default=1,
                        help='Stride for edge responses')
    parser.add_argument('--dilation', type=int, default=1,
                        help='Dilation for convolutional kernel')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='Use GPU for computation (if available)')
    args = parser.parse_args()
    return args

def plot_FMS(image, EHD_outputs, args):
    # Calculate the number of feature maps
    num_maps = EHD_outputs.size(1) + 1  # +1 for the original image
    
    # Calculate the number of columns (3 rows)
    num_cols = math.ceil(num_maps / 3)
    
    # Create a figure with larger size
    fig, axes = plt.subplots(3, num_cols, figsize=(6*num_cols, 18))
    axes = axes.flatten()  # Flatten the axes array for easier indexing
    
    # Plot the original image
    axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Generate angle labels
    angles = np.arange(0, 360, args.angle_res)
    
    # Plot each feature map
    for i in range(EHD_outputs.size(1)):
        ax = axes[i+1]
        feature_map = EHD_outputs[0][i].detach().cpu().numpy()
        im = ax.imshow(feature_map, cmap='viridis')
        if i == EHD_outputs.size(1) - 1:
            ax.set_title('No Edge')
        else:
            ax.set_title(f"{angles[i]}°")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Remove any unused subplots
    for i in range(num_maps, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

#Example for EHD feature
if __name__ == "__main__":
    
    #Load parameters for EHD
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    #load image
    image_path = 'C:/Users/aksha/Peeples_Lab/SEM_Images_Lacunarity/Datasets/Lung Cells SEM Images_group1_DC_NEW/Lung Cells Exposed to Crystalline Silica (CS)/CS_Bottom of Insert_DC_02052021_5KV_2500_ETD-HV_10.png'
    image_edges = Image.open(image_path)
    image_edges = np.array(image_edges)
    X = (image_edges/image_edges.max())*255
    
    #Convert to Pytorch tensor (Batch x num channels x height x width)
    X = (torch.tensor(X).unsqueeze(0).unsqueeze(0)).double()
    
    #Compute EHD feature
    EHD_features = Get_EHD(X, args, device=None)
    
    #Visualize edge responses
    plot_FMS(X.squeeze(0).squeeze(0), EHD_features, args)
    
    
    
    
    
    
    
    
    
    
    