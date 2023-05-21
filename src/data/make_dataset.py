import os
import requests
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from src.features.build_features import build_features


class ABCDataset(Dataset):
    def __init__(self, data_type, device="cpu"):
        dataset = h5py.File(f"./data/processed/{data_type}.h5", 'r')

        self.features = torch.from_numpy(dataset["features"][:]).to(device)
        self.targets = torch.from_numpy(dataset["targets"][:]).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx].unsqueeze(0)

        return feature, target


def path_safe(func, path_idx, args):
    os.makedirs(os.path.dirname(args[path_idx]), exist_ok=True)
    return func(*args)

def augment_data(data):
    return np.concatenate([
        np.rot90(data, 0,[1,2]),            # Rotate 000, Flip Non
        np.rot90(data, 0,[1,2])[:,::-1,:],  # Rotate 000, Flip Ver
        np.rot90(data, 0,[1,2])[:,:,::-1],  # Rotate 000, Flip Hor
        np.rot90(data, 1,[1,2]),            # Rotate 090, Flip Non
        np.rot90(data, 1,[1,2])[:,::-1,:],  # Rotate 090, Flip Ver
        np.rot90(data, 1,[1,2])[:,:,::-1],  # Rotate 090, Flip Hor
        np.rot90(data, 2,[1,2]),            # Rotate 180, Flip Non
        np.rot90(data, 2,[1,2])[:,::-1,:],  # Rotate 180, Flip Ver
        np.rot90(data, 2,[1,2])[:,:,::-1],  # Rotate 180, Flip Hor
        np.rot90(data, 3,[1,2]),            # Rotate 270, Flip Non 
        np.rot90(data, 3,[1,2])[:,::-1,:],  # Rotate 270, Flip Ver
        np.rot90(data, 3,[1,2])[:,:,::-1],  # Rotate 270, Flip Hor
    ])

def make_dataset():
    data_types = ["train", "val", "test"]

    for data_type in data_types:
        # Source dataset from link
        with path_safe(open, 0, [f"./data/raw/{data_type}.h5", "wb"]) as f:
            data = requests.get(f"https://share.phys.ethz.ch/~pf/albecker/abc/09072022_1154_{data_type}.h5").content
            f.write(data)

        # Build features and augment data
        dataset = h5py.File(f"./data/raw/{data_type}.h5", 'r')
        features = build_features(dataset)
        targets = dataset["agbd"][:].astype("float64")
        if data_type == "train":
            features = augment_data(features)
            targets = np.tile(targets, 12) # Tile to expand targets along with features

        # Save processed dataset
        with path_safe(h5py.File, 0, [f"./data/processed/{data_type}.h5", 'w']) as f:
            f.create_dataset("features", data=features)
            f.create_dataset("targets", data=targets)

def main():
    make_dataset()

if __name__ == "__main__": main()
