# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:31:02 2020
@author: jpeeples
"""
from sklearn.manifold import TSNE
#from barbar import Bar
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from matplotlib import offsetbox
from Utils.Compute_FDR import Compute_Fisher_Score
import pdb
from sklearn.metrics import silhouette_score

def pass_image(x):
    return x

def eurosat_to_rgb(x):
    x = x[:, :, [3, 2, 1]].float()
    x.sub_(x.min())
    return x.div_(x.max())

def get_magnitude(x):
    x = x[:, :, 0].float()
    x.sub_(x.min())
    return x.div_(x.max())

def plot_components(data, proj, images=None, ax=None,
                    thumb_frac=0.05, cmap='copper'):
    # scaler = MinMaxScaler(feature_range=(0,255))

    if (images.shape[-1]> 3):
        reduce_func = eurosat_to_rgb

    elif (images.shape[-1] < 3):
        reduce_func = get_magnitude
    else:
        reduce_func = pass_image

    ax = ax or plt.gca()
    
    ax.plot(proj[:, 0], proj[:, 1], '.k')
    
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(reduce_func(images[i]),zoom=.2, cmap=cmap),
                                      proj[i])
            ax.add_artist(imagebox)
            
def Generate_TSNE_visual(dataloaders_dict,model,sub_dir,device,class_names,
                         histogram=True,Separate_TSNE=False):

        # Turn interactive plotting off, don't show plots
        plt.ioff()
        count=0
        
      #TSNE visual of (all) data
        #Get labels and outputs
        for phase in ['test']:
            GT_val = np.array(0)
            indices_train = np.array(0)
            model.eval()
            model.to(device)
            features_extracted = []
            saved_imgs = []
            for idx, (inputs, classes)  in enumerate(dataloaders_dict[phase]):
                images = inputs.to(device)
                labels = classes.to(device, torch.long)
                
                GT_val = np.concatenate((GT_val, labels.cpu().numpy()),axis = None)
                features = model(images)
                    
                features = torch.flatten(features, start_dim=1)
                
                features = features.cpu().detach().numpy()
                
                features_extracted.append(features)
                saved_imgs.append(images.cpu().permute(0,2,3,1).numpy())
                count+=images.shape[0]
                
                if count > 5000:
                    break
        
      
            features_extracted = np.concatenate(features_extracted,axis=0)
            saved_imgs = np.concatenate(saved_imgs,axis=0)
            
            #Compute FDR scores
            GT_val = GT_val[1:]
            FDR_scores, log_FDR_scores = Compute_Fisher_Score(features_extracted,GT_val)
            np.savetxt((sub_dir+'{}_FDR.txt'.format(phase)),FDR_scores,fmt='%.2E')
            np.savetxt((sub_dir+'{}_log_FDR.txt'.format(phase)),log_FDR_scores,fmt='%.2f')
            features_embedded = TSNE(n_components=2,verbose=1,init='random',
                                     random_state=42).fit_transform(features_extracted)
            num_feats = features_extracted.shape[1]
        
            fig6, ax6 = plt.subplots()
            colors = colormap.rainbow(np.linspace(0, 1, len(class_names)))
            for texture in range (0, len(class_names)):
                x = features_embedded[[np.where(GT_val==texture)],0]
                y = features_embedded[[np.where(GT_val==texture)],1]
                
                ax6.scatter(x, y, color = colors[texture,:],label=class_names[texture])
             
            plt.title('TSNE Visualization of {} Data Features'.format(phase.capitalize()))
            box = ax6.get_position()
            ax6.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            ax6.legend(loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True, ncol=1)
            plt.subplots_adjust(right=0.8)
            plt.axis('off')
            
            fig6.savefig((sub_dir + 'TSNE_Visual_{}_Data.png'.format(phase.capitalize())), 
                         dpi=fig6.dpi, bbox_inches="tight")
            plt.close()
    
       
        # del dataloaders_dict,features_embedded
        torch.cuda.empty_cache()
        
        return FDR_scores, log_FDR_scores
