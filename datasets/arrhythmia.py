import os, scipy.io 

import numpy as np

from datasets.base import BaseDataset

class ArrhythmiaDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = 'arrhythmia'
        self.tmp_file_names = ['arrhythmia.mat']

    def load(self,):

        data = scipy.io.loadmat(os.path.join(self.data_path, 'arrhythmia.mat'))
        self.data_table  = data['X']  
        self.target = ((data['y']).astype(np.int32)).reshape(-1)

        self.norm_samples = self.data_table [self.target == 0]
        self.anom_samples = self.data_table [self.target == 1]

        self.norm_samples = np.c_[self.norm_samples, 
                                  np.zeros(self.norm_samples.shape[0])]

        self.anom_samples = np.c_[self.anom_samples, 
                                  np.ones(self.anom_samples.shape[0])]
        
        self.ratio = (100.0 * (0.5*len(self.norm_samples)) /
                     ((0.5*len(self.norm_samples)) + len(self.anom_samples)))

        self.data_table = np.concatenate((self.norm_samples, self.anom_samples),
                                         axis=0)

        self.N, self.D = self.data_table.shape
        self.D -= 1

        self.cat_features = [1, 21, 22, 23, 24, 25, 26]
        self.num_features = [x for x in list(range(0, self.D)) if x not in self.cat_features]
            
        self.cardinalities = [(1,2),(21,2),(22,11),(23,26),
                            (24,20),(25,7),(26,3)]
        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}

        print(f'There are {self.D} feature in total, with {len(self.num_features)}'
              f'continuous features and {len(self.cat_features)} categorical features')
        
        self.is_data_loaded = True

    def __repr__(self):
        repr = f'ArrhythmiaDataset(BaseDataset): {self.N} samples, {self.D} features\n'\
               f'{len(self.cat_features)} categorical features\n'\
               f'{len(self.num_features )} numerical features'
        return repr
        

