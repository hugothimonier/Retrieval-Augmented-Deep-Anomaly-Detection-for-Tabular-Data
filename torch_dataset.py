from torch.utils.data import Dataset
import torch

import numpy as np 

from utils.encode_utils import encode_data

class TorchDataset(Dataset):

    def __init__(self, dataset, mode, kwargs, device):

        ## encoding data:
        ## at this point numerical features have been normalized
        ## keep numerical columns as such
        ## transforming categorical features into one-hot encoding
        self.train = dataset.train[:,:-1]
        self.val, self.val_label = dataset.val[:,:-1], dataset.val[:,-1]
        self.n_features = dataset.D
        self.dataset_name = dataset.name
        self.cardinalities = dataset.cardinalities
        self.num_or_cat = dataset.num_or_cat
        self.ratio = dataset.ratio

        self.train, self.val = encode_data(self.train, self.val, dataset.D, dataset.cardinalities)
        self.device = device
        
        self.train = [torch.from_numpy(x) for 
                              x in self.train]
        self.val = [torch.from_numpy(x) for 
                             x in self.val]
        self.val_label = torch.as_tensor(self.val_label, 
                                         dtype=torch.int)
        
        ## At this point data structure is as follows:
        ## a list of columns (i.e. features), each column
        ## is of dimension N x 1 if numerical
        ## or  N x cardinality if categorical (one-hot encoded)
        self._mode = mode
        self.ratio = dataset.ratio
        self.full_dataset_cuda = kwargs.full_dataset_cuda

        if self._mode == 'train':
            self.batch_learning = (kwargs.exp_batch_size != -1)
        else:
            self.batch_learning = (kwargs.exp_val_batchsize != -1)
        self.retrieval = (kwargs.exp_retrieval_type != 'None')
        self.num_train_inference = kwargs.exp_retrieval_num_candidate_helpers

        if self.full_dataset_cuda or not self.batch_learning:
            if self._mode=='train':
                self.train = [x.to(self.device) for 
                             x in self.train]
            elif self._mode=='val':
                self.train = [x.to(self.device) for 
                              x in self.train]
                self.val = [x.to(self.device) for 
                             x in self.val]
                self.val_label = self.val_label.to(self.device)
            

        assert self.num_train_inference<len(self.train[0])+1, ('Number of training '
                                                               'samples for inference '
                                                               'cannot be higher than '
                                                               'training set length.')

    def __len__(self,):
        if self._mode=='train':
            return len(self.train[0])
        if self._mode=='val':
            return len(self.val[0])
        
    def __getitem__(self, index):

        if self._mode=='train':
            return [x[index].to(self.device) for x in self.train]
        
        elif self._mode=='val':
            if not self.full_dataset_cuda:
                val_batch = [x[index].to(self.device) for x in self.val]
                val_batch_label = self.val_label[index].to(self.device)
            else:
                val_batch = [x[index] for x in self.val]
                val_batch_label = self.val_label[index]
            return (val_batch, val_batch_label)
                    
    def set_mode(self, mode):
        assert mode in ['train', 'val']
        self._mode = mode
        if self.full_dataset_cuda or not self.batch_learning:
            self.put_on_device(mode)

    def put_on_device(self, mode):
        if self._mode=='train':
            self.train = [x.to(self.device) for 
                             x in self.train]
        if self._mode=='val':
            self.train = [x.to(self.device) for 
                              x in self.train]
            self.val = [x.to(self.device) for 
                             x in self.val]
            self.val_label = self.val_label.to(self.device)



        


        

        