import torch
from src.models.vit import ViT
from src.features.build_features import load_dataset

torch.manual_seed(0)

model = ViT(15, 5, 4, n_enc_blocks=2)
train_images, train_agbd = load_dataset('train')
val_images, val_agbd = load_dataset('val')

model.fit(train_images, train_agbd,
          val_images, val_agbd,
          epochs=30,
          batch_size=1138,
          lr=1e-2
)

