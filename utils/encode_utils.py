import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
#from category_encoders import *

def encode_data(train_data:np.ndarray,
                val_data:np.ndarray,
                num_features,
                cardinalities,):

    train_encoded_dataset = []
    val_encoded_dataset = []
    categorical_idx = [card[0] for card in cardinalities]
    
    for col_index in range(num_features):
        train_col = train_data[:,col_index].reshape(-1, 1)
        val_col = val_data[:,col_index].reshape(-1, 1)
        if col_index in categorical_idx:
            concat_col = np.concatenate([train_col, val_col])
            fitted_encoder = OneHotEncoder(sparse=False).fit(concat_col)
            encoded_train_col = fitted_encoder.transform(train_col).astype(np.float32)
            encoded_val_col = fitted_encoder.transform(val_col).astype(np.float32)
            train_encoded_dataset.append(np.array(encoded_train_col))
            val_encoded_dataset.append(np.array(encoded_val_col))
        else:
            train_encoded_dataset.append(np.array(train_col).astype(np.float32))
            val_encoded_dataset.append(np.array(val_col).astype(np.float32))
    return train_encoded_dataset, val_encoded_dataset


def torch_cast_to_dtype(obj, dtype_name):
    if dtype_name == 'float32':
        obj = obj.float()
    elif dtype_name == 'float64':
        obj = obj.double()
    elif dtype_name == 'long':
        obj = obj.long()
    else:
        raise NotImplementedError

    return obj

def get_numpy_dtype(dtype_name):
    if dtype_name == 'float32':
        dtype = np.float32
    elif dtype_name == 'float64':
        dtype = np.float64
    else:
        raise NotImplementedError

    return dtype


def get_torch_dtype(dtype_name):
    if dtype_name == 'float32':
        dtype = torch.float32
    elif dtype_name == 'float64':
        dtype = torch.float64
    else:
        raise NotImplementedError

    return dtype