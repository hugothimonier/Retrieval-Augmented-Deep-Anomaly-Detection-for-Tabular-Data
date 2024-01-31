import os

import pandas as pd
import numpy as np

from datasets.base import BaseDataset

class EcoliDataset(BaseDataset):

    '''
    https://archive.ics.uci.edu/ml/datasets/ecoli
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.is_data_loaded = False
        self.tmp_file_names = ['ecoli.data']
        self.name = 'ecoli'

    def load(self,):
        
        filename = os.path.join(self.data_path, self.tmp_file_names[0])
        data = pd.read_csv(filename, header=None, sep='\s+')

        self.anom_samples = data[data[8].isin(['omL','imL','imS'])]
        self.norm_samples = data[~data[8].isin(['omL','imL','imS'])]

        self.anom_samples.drop([8,0], axis=1, inplace=True)
        self.norm_samples.drop([8,0], axis=1, inplace=True)

        self.norm_samples = np.c_[self.norm_samples, 
                            np.zeros(self.norm_samples.shape[0])]
        self.anom_samples = np.c_[self.anom_samples, 
                                  np.ones(self.anom_samples.shape[0])]

        self.ratio = (100.0 * (0.5*len(self.norm_samples)) / ((0.5*len(self.norm_samples)) +
                                                             len(self.anom_samples)))
        self.data_table = np.concatenate((self.norm_samples, self.anom_samples),
                                         axis=0)
        
        self.N, self.D = self.data_table.shape
        self.D -= 1
        self.num_features = list(range(self.D))
        self.cat_features = []
        self.cardinalities = []
        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}

        self.anom_samples = self.anom_samples.astype(float)
        self.norm_samples = self.norm_samples.astype(float)

        self.is_data_loaded = True

    def __repr__(self):
        repr = f'EcoliDataset(BaseDataset): {self.N} samples, {self.D} features\n'\
               f'{len(self.cat_features)} categorical features\n'\
               f'{len(self.num_features)} numerical features'
        return repr