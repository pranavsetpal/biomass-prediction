import torch
from src.data.make_dataset import ABCDataset, path_safe
from src.models.model import ViT

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ABCDataset("train", device)
val_dataset = ABCDataset("val", device)

_, image_size, n_features = train_dataset[0][0].shape
patch_size = 5

params = {
  "n_enc_blocks": [1,2,3,4],
  "batch_size": [1138, 9104, 25036, 75108],
  "lr": [0.01, 0.005, 0.001]
}

best_model = None
best_loss = float('inf')

for n_enc_blocks in params["n_enc_blocks"]:
  for batch_size in params["batch_size"]:
    for lr in params["lr"]:
      model = ViT(image_size, patch_size, n_features, n_enc_blocks=n_enc_blocks).to(device)
      loss = model.fit(train_dataset,
        val_dataset,
        epochs=50,
        batch_size=batch_size,
        lr=lr
      )

      if loss < best_loss:
        best_model = model
        best_loss = loss

print("Best model:", best_model)
print("Best loss: ", best_loss)

path_safe(torch.save, 1, [best_model.state_dict(), "./models/vit.pt"])
