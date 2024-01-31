import numpy as np
import random, math, itertools
from typing import List 

import torch

def generate_mask_train(data_shape:tuple,
                        p_mask:float=0.15,
                        force_mask:bool=False
                        )->torch.Tensor:
    '''
    Apply bert-style masking:
        - we select a substet of features to be masked given a masking
        probability p_mask. Each feature for each sample has probability
        p_mask to be masked.
        - A masked feature will lead to different mask generation depending
        on the feature type since embedding is processed differently for
        each feature type. 
            - numerical data: for a vanilla linear embedding producing for
            each feature a d-dimensional vector (...)

    Args:
    - data_shape: tuple containing the data shape n x d.
    - hidden_dim: the chosen dimension for the embedding of numerical
                   features.
    - cardinalities: a list of tuples containing (idx, card), the index
                     and corresponding cardinality of each of the catego-
                     -rical features.
    - p_mask: the masking probability of each feature.

    Return:
    The mask matrix, a tensor boolean. 
    '''

    ## mask : 0, unmasked: 1

    n,d = data_shape
    mask_matrix_idx = torch.ones((n,d), dtype=torch.bool).flatten()
    Nm = len(mask_matrix_idx)
    
    std = np.sqrt(Nm * p_mask * (1 - p_mask))
    # This gives the total number of masks sampled in our approximative
    # sampling scheme as.
    num_masks_sampled = int(
        p_mask * Nm +
        np.random.normal(0, std))
    
    # We now take a random subset of the total number of mask indices.
    mask_indices_indices = np.random.choice(
            np.arange(0, Nm),
            size=num_masks_sampled,
            replace=False)
    
    mask_matrix_idx[mask_indices_indices] = 0
    mask_matrix_idx = mask_matrix_idx.reshape(n,d)

    if force_mask:
        # at this point we want to make sure that at least one element is masked per row
        row_has_zero = torch.any(mask_matrix_idx == 0, dim=1)
        # Identify rows without 0s
        rows_without_zero = torch.nonzero(~row_has_zero).view(-1)
        # Iterate through rows without 0s and replace a random '1' with '0'
        for row_index in rows_without_zero:
            row = mask_matrix_idx[row_index]
            ones_indices = torch.nonzero(row).view(-1)
            if len(ones_indices) > 0:
                random_index = random.choice(ones_indices)
                mask_matrix_idx[row_index, random_index] = 0
    return mask_matrix_idx

def generate_mask_val(data_shape:tuple,
                      n_hidden_features:List=[],
                      val_batchsize:int=0,
                      deterministic_mask:bool=True,
                      num_masks_val:int=0,
                      p_mask_val:float=0.15,
                      )->List:
    '''
    We generate all mask vector that mask
    simulatenously x features at a time, for each x in
    n_hidden_features. So if we have 
    n_hidden_features=[1,2]
    we consider each mask that masks 1 and 2 features at a time.
    Args:
    - data_shape: tuple containing the data shape (n,D)
    - n_hidden_features: list of number of features to be
                            masked simultaneously.
    - val_batchsize: how many validation samples at a time we 
                         consider.
    Return:
    A list of tensor masks (boolean tensors).
    '''
    augmentation_mask_matrices = []
    n,D = data_shape
    num_reconstruction = 0
    if deterministic_mask:
        n_hidden_features = list(set(n_hidden_features))
        n_hidden_features = [x for x in n_hidden_features
                                       if (x < D-1) & (x > 0)]
        for n_feature in sorted(n_hidden_features):
            num_reconstruction += math.comb(D, n_feature)
            for ele in itertools.combinations(range(D), n_feature):
                mask = torch.zeros(D, dtype=torch.bool)
                mask[ele,] = 1

                mask_val_matrix = (torch.stack([mask]*val_batchsize) 
                                    if val_batchsize>1 else mask.unsqueeze(0))
                augmentation_mask_matrices.append(mask_val_matrix)

    else:
        n_iter = 0
        incr = 0.05
        augmentation_mask_matrices_set = set()
        while len(augmentation_mask_matrices_set)<num_masks_val:
            n_iter += 1
            if n_iter > 200:
                print(f'Augmenting p_mask from {p_mask_val} to {p_mask_val+incr}',
                     'to converge.')
                p_mask_val += incr
                n_iter = 0
            
            mask = (torch.rand(D) > p_mask_val).bool()
            mask_val_matrix = (torch.stack([mask]*val_batchsize) 
                               if val_batchsize>1 else mask.unsqueeze(0))
            if mask_val_matrix not in augmentation_mask_matrices_set:
                augmentation_mask_matrices_set.add(mask_val_matrix)
                augmentation_mask_matrices.append(mask_val_matrix)

    return augmentation_mask_matrices


def apply_mask(original_data:List[torch.Tensor],
               cardinalities:dict,
               p_mask:float=0.15,
               force_mask:bool=False,
               device:torch.device=None,
               eval_mode:bool=False,
               mask_matrix:torch.Tensor=None,
               is_distributed:bool=False
               )-> dict:
    datashape = (len(original_data[0]),len(original_data))
    if not eval_mode:
        mask_matrix = generate_mask_train(datashape,
                                          p_mask,
                                          force_mask).to(device)
    else:
        mask_matrix = mask_matrix.to(device)
        
    mask_matrix_list = [mask_matrix[:datashape[0],idx].unsqueeze(1) for idx in range(mask_matrix.shape[1])]

    for idx, card in cardinalities:
        # change mask dimension to fit one-hot encoding to easily obtain masked_tensor
        mask_matrix_list[idx] = mask_matrix_list[idx].view(datashape[0],1).expand(datashape[0], card)
    
    if is_distributed:
        #useful for val only
        mask_matrix_list = [mask[:datashape[0],:] for mask in mask_matrix_list]
    
    mask_matrix_list = [mask.to(device) for mask in mask_matrix_list]
    original_data = [x.to(device) for x in original_data]
    masked_tensor = [x * mask for x, mask in zip(original_data, mask_matrix_list)]

    #restore the mask dimension for the categorical features
    for idx, card in cardinalities:
        # change mask dimension to fit one-hot encoding to easily obtain masked_tensor
        mask_matrix_list[idx] = mask_matrix_list[idx][:,0].unsqueeze(1)
        
    # at this point mask: 0, unmasked: 1. We invert from now.
    masked_tensor = [torch.cat((x, ~mask_matrix_list[idx]), dim=1)
                     for idx, x in enumerate(masked_tensor)]
    
    mask_matrix_list = [~x for x in mask_matrix_list]

    ## in masked_tensor, first dimension corresponds to the scalar + mask token for numerical 
    ## or one-hot + mask_token for categorical.
    ## at this point mask_token == 1 <=> masked, mask_token==0 <=> unmasked

    data_dict = {
        'masked_tensor': masked_tensor,
        'ground_truth': original_data,
        'mask_matrix': ~mask_matrix.to(device),
        'mask_matrix_list': mask_matrix_list
    }

    return data_dict


    

    
    

    




