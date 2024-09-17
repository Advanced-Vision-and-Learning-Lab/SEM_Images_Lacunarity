# -*- coding: utf-8 -*-
"""
Parameters for lacunarity experiments
"""

def Parameters(args):
    
    ######## ONLY CHANGE PARAMETERS BELOW ########
    #Flag for if results are to be saved out
    #Set to True to save results out and False to not save results
    save_results = args.save_results
    
    folder = args.folder
    texture_feature_ = args.texture_feature
    texture_feature_names = {1:'Fractal_Dimension', 2:'Base_Lacunarity', 3: "GAP"}
    texture_feature = texture_feature_names[texture_feature_]

    agg_func_selection = args.agg_func
    agg_func_names = {1:'global', 2:'local'}
    agg_func = agg_func_names[agg_func_selection]
    
    #Select dataset
    data_selection = args.data_selection
    Dataset_names = {1: 'LungCells_DC'}
    
    #Lacunarity Parameters
    kernel = args.kernel
    stride = args.stride
    padding = args.padding
    quant_levels = args.quant_levels
        
    #Location of texture datasets
    Data_dirs = {'LungCells_DC': 'Datasets/Lung Cells SEM Images_group1_DC_NEW'}
    
    #channels in each dataset
    channels = {'LungCells_DC': 1}
    
    #Number of classes in each dataset
    num_classes = {'LungCells_DC': 3}
    
    Dataset = Dataset_names[data_selection]
    data_dir = Data_dirs[Dataset]
    
    #Return dictionary of parameters
    Params = {'save_results': save_results,'folder': folder,
              'texture_feature': texture_feature, 'agg_func': agg_func,
            'Dataset': Dataset, 'data_dir': data_dir,
            'kernel': args.kernel, 'stride': args.stride, 
            'quant_levels':quant_levels, 'conv_padding': args.padding,
            'num_classes': num_classes, 'channels': channels}
    
    return Params