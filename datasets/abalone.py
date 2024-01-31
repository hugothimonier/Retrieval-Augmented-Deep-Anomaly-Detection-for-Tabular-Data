import os

import pandas as pd
import numpy as np

from datasets.base import BaseDataset

class AbaloneDataset(BaseDataset):
    '''
    https://archive.ics.uci.edu/ml/datasets/abalone
    '''
    def __init__(self, **kwargs):
        super(AbaloneDataset, self).__init__(**kwargs)

        self.file_name = ['abalone.data']
        self.name = 'abalone'

    def load(self):
        data = pd.read_csv(os.path.join(self.data_path, 
                    self.file_name[0]),
                    header=None, sep=',')
        data = data.rename(columns={8: 'y'})
        data['y'].replace([8, 9, 10], 0, inplace=True)
        data['y'].replace([3, 21], 1, inplace=True)
        data.iloc[:, 0].replace('M', 0, inplace=True)
        data.iloc[:, 0].replace('F', 1, inplace=True)
        data.iloc[:, 0].replace('I', 2, inplace=True)

        self.target = ((np.array(data['y'])).astype(np.int32)).reshape(-1)
        self.data_table  = data.loc[:, data.columns != 'y'].to_numpy()

        self.norm_samples = self.data_table [self.target == 0]
        self.anom_samples = self.data_table [self.target == 1]

        # at the moment the label is added as the last column (maybe should be removed)
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

        self.cat_features = [0]
        self.num_features = list(range(1, self.D))
        self.cardinalities = [(0,3)]
        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}

        self.is_loaded = True

    def __repr__(self):
        repr = f'AbaloneDataset(BaseDataset): {self.N} samples, {self.D} features\n'\
               f'{len(self.cat_features)} categorical features\n'\
               f'{len(self.num_features)} numerical features'
        return repr