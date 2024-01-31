import os
from operator import itemgetter

import pandas as pd
import numpy as np

from datasets.base import BaseDataset

SHIFT_TO_YEAR = {'iid':[2006,2007,2008,2009,2010],
                'near':[2011,2012,2013],
                'far':[2014,2015]}

class AnoshiftDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

        self.num_target_cols = []
        self.is_data_loaded = False
        self.tmp_file_names = ['arrhythmia.mat']

        self.ad = True
        self.fixed_test_set_index = None
        self.shift = c.shift

        assert self.c.shift in ['iid', 'near','far']

    def load(self):

        categorical_cols = ['0', '1', '2', '3', '13']
        numerical_cols = ['4', '5', '6', '7', '8', '9', '10', '11', '12']
        additional_cols = ['14', '15', '16', '17', '19']
        label_col = ['18']

        train_df = self.load_train()
        test_df = self.load_test()

        #train_target = train_df[label_col]
        #test_target = test_df[label_col]

        train_df = pd.concat([train_df.loc[:,categorical_cols], 
                                train_df.loc[:,numerical_cols], train_df.loc[:,label_col]], axis=1)

        test_df = pd.concat([test_df.loc[:,categorical_cols], 
                                test_df.loc[:,numerical_cols], test_df.loc[:,label_col]], axis=1)

        for col in numerical_cols + label_col:
            train_df[col] = pd.to_numeric(train_df[col])
            test_df[col] = pd.to_numeric(train_df[col])

        # change label value from 1: normal, -1,-2:anomaly
        # to 0: normal, 1: anomaly
        for df in [train_df, test_df]:
            df[label_col] = df[label_col].replace(1,0)
            df[label_col] = df[label_col].replace([-1,-2],1)

        self.train_index = train_df.shape[0]

        self.D = train_df.shape[1]
        self.N = train_df.shape[0] + test_df.shape[0]

        self.cat_features = list(range(0,5))
        self.num_features = list(range(5, self.D - 1))
        self.cat_target_cols = [self.D] #last column is target

        self.data_table = np.concatenate([np.array(train_df), np.array(test_df)], axis=0)

        self.is_data_loaded = True

    def load_train(self):
        
        pdList = list()
        year= 2006
        while year <= 2010:
            df = pd.read_parquet(os.path.join(self.c.data_path, f'subset/{year}_subset.parquet'))
            df = df.reset_index(drop=True)
            pdList.append(df)
            year += 1

        return pd.concat(pdList)

    def load_test(self):

        pdList = list()
        for year in SHIFT_TO_YEAR[self.shift]:
            if year <= 2010:
                df = pd.read_parquet(os.path.join(self.c.data_path, f'subset/{year}_subset_valid.parquet'))
            else:
                df = pd.read_parquet(os.path.join(self.c.data_path, f'subset/{year}_subset.parquet'))
            df = df.reset_index(drop=True)
            pdList.append(df)
        
        return pd.concat(pdList)
