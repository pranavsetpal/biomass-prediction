import torch
from src.models.vit import ViT
from src.features.build_features import load_dataset

torch.manual_seed(0)

train_features, train_target = load_dataset('train')
val_features, val_target = load_dataset('val')

_, _, image_size, n_features = train_features.shape
patch_size = 3

model = ViT(image_size, patch_size, n_features, n_enc_blocks=4)
model.fit(train_features, train_target,
          val_features, val_target,
          epochs=30,
          batch_size=1138,
          lr=0.01
)
