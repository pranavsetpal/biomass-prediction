import numpy as np

def build_features(dataset):
    veg_index = (dataset['images'][:,:,:,[8]] - dataset['images'][:,:,:,[4]]) /\
                (dataset['images'][:,:,:,[8]] + dataset['images'][:,:,:,[4]] + 1e-08)

    features = np.concatenate([
        dataset['images'][:,:,:,[1,7,10]] / 10000,
        veg_index
    ], 3)

    return features
