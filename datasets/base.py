
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np 

class BaseDataset():

    def __init__(self, **kwargs):
        self.data_path = kwargs['data_path']
        self.D = None
        self.N = None
        self.cat_features = []
        self.num_features = []
        self.cat_as_num = kwargs['exp_cat_as_num']
        ## list composed of the cardinality of each cat feature
        ## probably in the form of a tuple: (idx, cardinality)
        self.cardinalities = []
        self.is_loaded = False

    def load(self):
        return None
    
    def normalize(self, train:np.array, val:np.array):

        if len(self.num_features)>0:
            scaler = MinMaxScaler()
            scaler.fit(train[:,self.num_features])
            train[:,self.num_features] = scaler.transform(train[:,self.num_features])
            val[:,self.num_features] = scaler.transform(val[:,self.num_features])

        return train, val

    def split_train_val(self,
                        seed:int,
                        iteration:int,
                        contamination_share:float=0.):

        train, val_norm = train_test_split(self.norm_samples,
                                                test_size=0.5,
                                                random_state=seed+iteration,
                                                shuffle=True)
        if contamination_share==0:
            val = np.concatenate((val_norm, self.anom_samples))
            self.train, self.val = self.normalize(train, val)
        else:
            # for comparability, even though we only take a number of anomalies
            # to get a share of 1% in the training set, we always hold out
            # 10% of anomalies so that the validation set's share is not altered.
            ten_perc_anom = compute_num_anom(0.1, len(train))
            self.hold_out_anom, self.anom_samples = split_array(self.anom_samples, 
                                                               ten_perc_anom)
            num_train_anom = compute_num_anom(contamination_share,
                                              len(train))
            random_idx = np.random.choice(self.hold_out_anom.shape[0],
                                          num_train_anom,
                                          replace=False)
            anom_train = self.hold_out_anom[random_idx]

            train = np.concatenate((train, anom_train))
            np.random.shuffle(train)
            val = np.concatenate((val_norm, self.anom_samples))
            self.train, self.val = self.normalize(train, val)
            self.ratio = (100.0 * (0.5*len(self.norm_samples)) / ((0.5*len(self.norm_samples)) +
                                                             len(self.anom_samples)))
        print(f'There are {len(train)} samples in the training set.')
        print(f'There are {len(val)} samples in the validation set.')
            
            
def compute_num_anom(contamination_share, num_norm):
    number_anom = (contamination_share/(1 - contamination_share)) * num_norm
    return int(number_anom)

def split_array(array, n):
    # Shuffle the array randomly
    np.random.shuffle(array)
    
    # Take the first n elements for the first sub-array
    sub_array1 = array[:n]
    
    # The remaining elements form the second sub-array
    sub_array2 = array[n:]
    
    return sub_array1, sub_array2