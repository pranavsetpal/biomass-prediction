import torch
from src.models.model import ViT
from src.data.make_dataset import ABCDataset

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ABCDataset("train", device)
val_dataset = ABCDataset("val", device)

_, image_size, n_features = train_dataset[0][0].shape
patch_size = 5

model = ViT(image_size, patch_size, n_features, n_enc_blocks=2).to(device)
model.fit(train_dataset,
          val_dataset,
          epochs=2,
          batch_size=4552,
          lr=0.001
)
