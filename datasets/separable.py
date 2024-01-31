import os, scipy.io 
from operator import itemgetter

import pandas as pd
import numpy as np

from datasets.base import BaseDataset

class SeparableDataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.kwargs = kwargs
        self.num_target_cols = []
        self.is_data_loaded = False
        self.tmp_file_names = ['separable.npy']
        self.name = 'separable'

    def load(self, ):
        
        data = np.load(os.path.join(self.data_path, self.tmp_file_names[0]))

        self.data_table  = data[:,:-1]
        self.target = data[:,-1]

        self.norm_samples = self.data_table [self.target == 0]  # 1800 norm
        self.anom_samples = self.data_table [self.target == 1]  # 200 anom
        
        self.norm_samples = np.c_[self.norm_samples, 
                                  np.zeros(self.norm_samples.shape[0])]
        self.anom_samples = np.c_[self.anom_samples, 
                                  np.ones(self.anom_samples.shape[0])]
        
        self.ratio = (100.0 * (0.5*len(self.norm_samples)) / ((0.5*len(self.norm_samples)) +
                                                             len(self.anom_samples)))
        
        #self.anom_to_keep = np.c_[self.anom_samples[:100],
        #                          np.ones(self.anom_samples[:100].shape[0])]
        #self.anom_for_contamination = np.c_[self.anom_samples[100:],
        #                          np.ones(self.anom_samples[100:].shape[0])]

        #if self.kwargs.share_contamination>0.:
        #    self.num_anom_inference = compute_num_anom(self.kwargs.share_contamination,
        #                                         0.5 * len(self.norm_samples))
        #    max_share = len(self.anom_for_contamination) + (len(self.anom_for_contamination)+
        #                                                    0.5*len( self.norm_samples))
        #    err = f'share of anomalies is too high, has to be less or equal to {max_share}'
        #    assert self.num_anom_inference <= len(self.anom_for_contamination), err
        #    
        #    self.data_table = np.concatenate((self.anom_for_contamination, 
        #                        self.norm_samples,
        #                        self.anom_to_keep),
        #                        axis=0)

        #    self.ratio = 100.0 * (0.5*len(self.norm_samples)) / ((0.5*len(self.norm_samples)) +
        #                                                          len(self.anom_to_keep))

        #else:
        #    self.data_table = np.concatenate((self.norm_samples,
        #                          self.anom_to_keep),
        #                         axis=0)
        # 
        #     self.ratio = 100.0 * (0.5*len(self.norm_samples)) / ((0.5*len(self.norm_samples)) + 
        #                                                         len(self.anom_to_keep))

        self.N, self.D = self.data_table.shape
        self.D -= 1

        #here no categorical features
        self.cat_features = []
        self.num_features = list(range(0, self.D)) ##only numerical
        self.cardinalities = []
        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}

        self.is_data_loaded = True

    def __repr__(self):
        repr = f'SeparableDataset(BaseDataset): {self.N} samples, {self.D} features\n'\
               f'{len(self.cat_features)} categorical features\n'\
               f'{len(self.num_features)} numerical features'\
               f'Anomaly share: {self.kwargs.share_contamination}' 
        return repr

def compute_num_anom(contamination_share, num_norm):
    number_anom = (contamination_share/(1 - contamination_share)) * num_norm
    return int(number_anom)