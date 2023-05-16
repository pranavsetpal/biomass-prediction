import h5py
import numpy as np
import torch

def build_features(dataset):
    veg_index = (dataset['images'][:,:,:,[8]] - dataset['images'][:,:,:,[4]]) /\
                (dataset['images'][:,:,:,[8]] + dataset['images'][:,:,:,[4]] + 1e-08)

    features = torch.from_numpy(np.concatenate([
        dataset['images'][:,:,:,[1,7,10]] / 10000,
        veg_index
    ], 3))

    return features


def augment_data(data):
    return torch.cat([
        data.rot90(0,[1,2]),            # Rotate 000, Flip Non
        data.rot90(0,[1,2]).flip(1),    # Rotate 000, Flip Ver
        data.rot90(0,[1,2]).flip(2),    # Rotate 000, Flip Hor
        data.rot90(1,[1,2]),            # Rotate 090, Flip Non
        data.rot90(1,[1,2]).flip(1),    # Rotate 090, Flip Ver
        data.rot90(1,[1,2]).flip(2),    # Rotate 090, Flip Hor
        data.rot90(2,[1,2]),            # Rotate 180, Flip Non
        data.rot90(2,[1,2]).flip(1),    # Rotate 180, Flip Ver
        data.rot90(2,[1,2]).flip(2),    # Rotate 180, Flip Hor
        data.rot90(3,[1,2]),            # Rotate 270, Flip Non
        data.rot90(3,[1,2]).flip(1),    # Rotate 270, Flip Ver
        data.rot90(3,[1,2]).flip(2),    # Rotate 270, Flip Hor
    ])


def load_dataset(data_type):
    dataset = h5py.File(f"./data/{data_type}.h5", 'r')

    data = build_features(dataset)
    data = augment_data(data)

    target = torch.tensor(dataset['agbd'][:], dtype=torch.float64).unsqueeze(1)

    return data,target
