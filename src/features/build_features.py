import h5py
import torch

def load_dataset(data):
    dataset = h5py.File(f"./data/{data}.h5", 'r')

    images = dataset['images'][:,:,:,[0,1,7,10]]
    # Vegetation Index
    images[:,:,:,0] = (dataset['images'][:,:,:,8] - dataset['images'][:,:,:,4]) /\
                      (dataset['images'][:,:,:,8] + dataset['images'][:,:,:,4] + 1e-08)

    images = torch.from_numpy(images / 10000)

    agbd = torch.tensor(dataset['agbd'][:], dtype=torch.float64).unsqueeze(1)

    return images,agbd
