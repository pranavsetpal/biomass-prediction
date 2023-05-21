import torch
from src.data.make_dataset import ABCDataset
from src.models.model import ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = ABCDataset("test")
_, image_size, n_features = test_dataset[0][0].shape
patch_size = 5

model = ViT(image_size, patch_size, n_features, n_enc_blocks=2)
model.load_state_dict(torch.load("./models/vit.pt", map_location=device))

loss = model.evaluate(test_dataset)
print("Test RSME:", loss**(1/2))
