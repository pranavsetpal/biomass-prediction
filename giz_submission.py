import h5py
import torch
import pandas as pd
from src.features.build_features import build_features
from src.models.model import ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images_test_dataset = h5py.File("../africa-biomass-challenge/images_test.h5", 'r')
images_test = torch.from_numpy(build_features(images_test_dataset))
_, image_size, n_features = images_test[0].shape
patch_size = 5

model = ViT(image_size, patch_size, n_features, n_enc_blocks=2)
model.load_state_dict(torch.load("./models/vit.pt", map_location=device))

preds = model(images_test).squeeze().detach().numpy()

s2_id_pairs = pd.read_csv("../africa-biomass-challenge/UniqueID-SentinelPair.csv")

preds_pd = pd.DataFrame({"Target":preds}).rename_axis("S2_idx").reset_index()
preds_pd = s2_id_pairs.merge(preds_pd, on="S2_idx").drop(columns=["S2_idx"])

preds_pd.to_csv("../africa-biomass-challenge/GIZ_Biomass_predictions.csv", index=False)
