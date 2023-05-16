import h5py
import numpy as np
import torch

def load_dataset(data):
    dataset = h5py.File(f"./data/{data}.h5", 'r')

    veg_index = (dataset['images'][:,:,:,[8]] - dataset['images'][:,:,:,[4]]) /\
                (dataset['images'][:,:,:,[8]] + dataset['images'][:,:,:,[4]] + 1e-08)

    features = np.concatenate([
        dataset['images'][:,:,:,[1,7,10]] / 10000,
        veg_index
    ], 3)
    features = torch.from_numpy(features)

    target = torch.tensor(dataset['agbd'][:], dtype=torch.float64).unsqueeze(1)

    return features,target
