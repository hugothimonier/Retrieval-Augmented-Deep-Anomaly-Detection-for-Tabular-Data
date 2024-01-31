import sys
from typing import List
from collections import defaultdict

import torch
import torch.nn as nn

from utils.encode_utils import torch_cast_to_dtype

class Loss:
    '''
    Compute losses.
    '''

    def __init__(self, args, is_batchlearning,
                 device=None,):
        
        self.args = args
        self.is_batchlearning = is_batchlearning
        self.device = device
        
        self.cross_ent_loss = nn.CrossEntropyLoss(reduction='sum')
        self.cross_ent_loss_no_sum = nn.CrossEntropyLoss(reduction='none')
        self.loss_stats = [
                            'num_loss', 'num_total_preds',
                            'cat_loss', 'total_loss', 'cat_total_preds'
                           ]
        self.loss_modes = ['train','val']
        self.reset()

        self.loss_val = dict()
        self.normalized_loss_val = dict()
        self.current_loss_val = dict()

        self.val_loss_logg = {'num_val_pred':0, 
                              'loss_val_epoch':torch.tensor([0.],
                                                device=self.device),
                             }
        self._mode = 'train'
        self.val_losses = []

    def set_mode(self, mode:str):
        print(f'Setting loss mode to {mode}.')
        self._mode = mode
        
    def reset(self):
        """Reset batch and epoch losses."""
        self.batch_loss = None
        self.epoch_loss = None
        
    def reset_logs(self):
        """Reset value stored for val loggs"""
        self.val_loss_logg = {'num_val_pred':0, 
                              'loss_val_epoch':torch.tensor([0.], device=self.device),
                              }
        
    def update_losses(self,):
        """Update losses.

        In the case of batchlearning SGD, this function should only be
         called after we have already backpropagated on the batch loss,
         because we detach all those tensors prior to adding them to
         the epoch loss (no need to retain the computation graph if
         we are just logging per-epoch).

        Set batch loss to current value.

        Add current batch loss to epoch loss.
        """
        # Throw away gradient information:
        #   (I) if we are evaluating the model, or
        #   (II) if we are batchlearning (because we have already
        #           backpropped, and just want to store loss info for per-epoch
        #           logging).
        if self._mode=='val' or self.is_batchlearning:
            self.detach_all(self.batch_loss)

        # update epoch loss to incorporate current batch
        if self.epoch_loss is None:
            self.epoch_loss = self.batch_loss
        else:
            for mode in self.epoch_loss.keys():
                for key in self.epoch_loss[mode].keys():
                    self.epoch_loss[mode][key] = (
                        self.epoch_loss[mode][key] +
                        self.batch_loss[mode][key])


    def compute(self, *args, **kwargs):
        """Compute loss and update batch losses."""

        loss_dict = self.compute_loss(*args, **kwargs)
        self.batch_loss = loss_dict

    def compute_loss(self, 
                     output:List[torch.Tensor], 
                     ground_truth_data:List[torch.Tensor],
                     num_or_cat:dict,
                     mask_matrix:torch.Tensor):
        '''
        Compute total loss for prediction.
        Args:
        - output (list of torch.Tensor): prediction of the model in the form
                                         of a list of tensors.
        - ground_truth_data (list of torch.Tensor): true value of the columns
                                                    in the form of a list of 
                                                    tensors.
        - num_or_cat (dict): dictionnary indicating for each index if it corresponds
                             to a categorical feature.
        - mask_matrix (torch.Tensor) 

        Returns:
        loss for batch.
        '''
        # Compute losses per column
        loss_dict = dict()
        loss_dict = {
            loss_mode: {
                key: torch.zeros((1), device=self.device)
                if 'loss' in key
                else 0 for key in self.loss_stats
            }
            for loss_mode in self.loss_modes
        }

        for col, (out, dat) in enumerate(zip(output, ground_truth_data)):
            is_cat = (not num_or_cat[col])
            col_mask = mask_matrix[col].squeeze(1)
            num_preds = col_mask.sum()

            # Compute loss for selected row entries in col
            loss = self.compute_column_loss(
                col=col, is_cat=is_cat, 
                output=out, data=dat,
                col_mask=col_mask,)
            
            if is_cat:
                loss_dict[self._mode]['cat_loss'] += loss
                loss_dict[self._mode]['cat_total_preds'] += num_preds
            else:
                loss_dict[self._mode]['num_loss'] += loss
                loss_dict[self._mode]['num_total_preds'] += num_preds
            loss_dict[self._mode]['total_loss'] += loss

        return loss_dict

    def compute_column_loss(self, col:int, is_cat:bool,
                            output:torch.Tensor,
                            data:torch.Tensor, 
                            col_mask:torch.Tensor):
        '''
        Compute loss for a column for selected rows.

        Args:
        - col (int): index of the current column
        - is_cat (bool): whether the current column is categorical or not.
        - output (torch.Tensor): model prediction.
        - data (torch.Tensor): true data.
        - eval_model (bool): whether the model is in eval_mode
        - col_mask (torch.Tensor): the column indicating whether the feature was masked.
                                   If so, the loss should be computed for that row.
        Returns:
        loss (torch.Tensor): Loss value for that column.
        '''

        if torch.isnan(output).any():
            print('Output is NaN, stopping training.')
            sys.exit(1)

        if is_cat:
            # Cross-entropy loss does not expect one-hot encoding.
            # Instead, convert data to integer list of correct labels:
            # long_data: list of ints, each entry in [0, ..., C-1]
            long_data = torch.argmax(
                torch_cast_to_dtype(obj=data, dtype_name=self.args.model_dtype),
                dim=1).to(device=self.device)
            
            # Compute sum of cross_entropy losses.
            loss = self.cross_ent_loss_no_sum(output, long_data)

            # Only count the loss for entries that were masked
            inter_loss = loss * col_mask

            if self._mode=='val':
                # implement here the storing of the validation loss
                # think about it first
                self.val_losses.append(inter_loss)

            # We use the unreduced loss above - reduce here
            loss = inter_loss.sum()

        else:
            # Apply the invalid entries multiplicatively, so we only
            # tabulate an MSE for the entries which were masked
            output = col_mask * output.squeeze()
            data = col_mask * data.squeeze()

            inter_loss = torch.square((output - data))
            # add here that we store in self.loss_val
            
            if self._mode=='val':
                # implement here the storing of the validation loss
                # think about it first
                self.val_losses.append(inter_loss)

            loss = torch.sum(inter_loss)

        return loss
    
    def get_individual_val_loss(self,):
        err = 'Mode should be validation and epoch run for the val set.'
        assert self._mode=='val' and len(self.val_losses)>0, err
        val_loss_mat = torch.stack(self.val_losses, dim=1)
        return torch.sum(val_loss_mat, dim=1).detach()
    
    def reset_val_loss(self,):
        self.val_losses = []
    
    def finalize_batch_loss(self,):
        """Normalise batch losses by number of predictions."""
        return self.finalize_losses(self.batch_loss, False)
    
    def finalize_epoch_losses(self, eval_model,):
        """Normalise epoch losses."""
        std_dict = self.finalize_losses(self.epoch_loss, eval_model)
        return std_dict
    
    def finalize_losses(self, raw_dict, eval_model):
        """Before we backpropagate or log, we need to finalise the losses.

        * calculate total loss by weighing label and augmentation losses and
            normalising by the total number of predictions made.
        * if we are evaluating model, also compute losses and accuracies for
            the 'label' and 'augmentation' categories separately

        We can only do this directly before backprop or logging, since only
        then do we know the total number of predictions, for example because
        we aggregate losses accumulated over several minibatches.

        """
        std_dict = defaultdict(dict)

        if eval_model:
            self.detach_all(raw_dict)

        # *** Logging Extra Losses ***
        # Has no bearing on backprop, just for logging purposes.
        for mode in raw_dict.keys():
            # keys are subset of {augmentation, label}
            # Normalize losses in the different modes and calculate accuracies.
            cat_preds = float(raw_dict[mode]['cat_total_preds'])
            num_preds = float(raw_dict[mode]['num_total_preds'])
            total_preds = cat_preds + num_preds

            if total_preds > 0:
                std_dict[mode]['total_loss'] = (
                    (raw_dict[mode]['total_loss'])
                    / total_preds)
            if num_preds > 0:
                std_dict[mode]['num_loss'] = raw_dict[mode]['num_loss'] / num_preds
            if cat_preds > 0:
                std_dict[mode]['cat_loss'] = raw_dict[mode]['cat_loss'] / cat_preds

        return std_dict
    
    def detach_all(self, raw_dict):
        # outer dict
        for mode in raw_dict.keys():
            for key, value in raw_dict[mode].items():
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    raw_dict[mode][key] = value.detach()