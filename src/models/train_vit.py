import torch
from src.models.vit import ViT
from src.features.build_features import load_dataset

torch.manual_seed(0)

model = ViT(15, 5, 4, n_enc_blocks=2)
train_features, train_target = load_dataset('train')
val_features, val_target = load_dataset('val')

model.fit(train_features, train_target,
          val_features, val_target,
          epochs=30,
          batch_size=1138,
          lr=1e-2
)
