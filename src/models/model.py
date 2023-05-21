import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Encoder(nn.Module):
    def __init__(self, n_patches, n_features, n_heads, dtype=torch.float64):
        super().__init__()

        self.layer_norm = nn.LayerNorm((n_patches, n_features), elementwise_affine=False, dtype=dtype)
        self.msa = nn.MultiheadAttention(n_features, n_heads, batch_first=True, dtype=dtype)
        self.mlp = nn.Sequential(
            nn.Dropout(),
            nn.Linear(n_features, n_features, dtype=dtype),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(n_features, n_features, dtype=dtype)
        )


    def forward(self, input):
        x = self.layer_norm(input)
        msa_output  = self.msa(x, x, x)[0]
        mlp_input   = msa_output + input
        mlp_output  = self.mlp(self.layer_norm(mlp_input))
        output      = mlp_output + mlp_input

        return output


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, n_channels, n_enc_blocks=1, dtype=torch.float64):
        super().__init__()

        n_patches = (image_size // patch_size)**2
        n_features = patch_size**2 * n_channels

        self.patch_size = patch_size

        self.class_token = nn.Parameter(torch.zeros(1, n_features))
        self.pos = nn.Parameter(self.pos_emb(n_patches+1, n_features))

        self.encoder = nn.ModuleList([
            Encoder(n_patches+1, n_features, patch_size, dtype)
            for _ in range(n_enc_blocks)
        ])

        self.linear = nn.Linear(n_features, 1, dtype=dtype)

        self.MSE = nn.MSELoss(reduction='sum')


    def forward(self, input):
        patches = self.patched(input, self.patch_size)

        patches = torch.cat([
            self.class_token.unsqueeze(0).repeat(len(patches),1,1),
            patches
        ], 1)
        enc_input = patches + self.pos

        for enc_block in self.encoder:
            enc_input = enc_block(enc_input)

        linear_input = enc_input[:,0]
        output = self.linear(linear_input)

        return output


    def loss(self, inputs, targets):
        preds = self(inputs)
        return self.MSE(preds, targets)

    def fit(self, train_dataset, val_dataset, epochs, batch_size, lr=1e-2):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        for epoch in range(epochs):

            train_loss = 0
            for X,Y in train_dataloader:
                optimizer.zero_grad()
                loss = self.loss(X, Y)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()
            train_loss /= len(train_dataset)

            val_loss = 0
            for X,Y in val_dataloader:
                val_loss += self.loss(X, Y).item()
            val_loss /= len(val_dataset)

            if (val_loss < 20):
                break
            print("---------------------")
            print("Num of epoches:", epoch)
            print("\tMSE  = ", val_loss)
            print("\tRMSE = ", val_loss**(1/2))
            print("---------------------")


    def patched(self, images, patch_size):
        # Input:  (n_samples, h_image, w_image, n_channels), (h_patch, w_patch)
        # Output: (n_samples, n_patches, n_channels*h_patch*w_patch)

        (n_samples, h_image, _, n_channels) = images.shape
        h_patch = w_patch = patch_size

        flattened_patches = (
            images.reshape(n_samples, h_image // h_patch, h_patch, -1, w_patch, n_channels) # Create patch segments
            .transpose(2, 3)                                                     # Join the segments, forming patch
            .reshape(n_samples, -1, n_channels*h_patch*w_patch)    # Remove excess dimensions and flatten the patch
        )
        return flattened_patches


    def pos_emb(self, n_patches, n_features):
        pe = torch.zeros((n_patches,n_features))

        i = d_patch = n_features
        pos = n_patches

        for pos in range(n_patches):
            for i in range(0,n_features,2):
                pe[pos,i] = np.sin(pos / 10000**(i/d_patch))
            for i in range(1,n_features,2):
                pe[pos,i] = np.cos(pos / 10000**(i/d_patch))

        return pe


class DeiT(ViT):
    def __init__(self, teacher_model, image_size, patch_size, n_channels, n_enc_blocks=1, dtype=torch.float64):
        super().__init__(image_size, patch_size, n_channels, n_enc_blocks, dtype)

        self.distillation_token = nn.Parameter(torch.zeros(1, n_features))
        self.pos = nn.Parameter(self.pos_emb(n_patches+2, n_features))
        self.linear = nn.Linear(n_features, 2, dtype=dtype)

        self.teacher_model = teacher_model


    def forward(self, input):
        patches = self.patched(input, self.patch_size)

        patches = torch.cat([
            self.class_token.unsqueeze(0).repeat(len(patches),1,1),
            self.distillation_token.unsqueeze(0).repeat(len(patches),1,1),
            patches
        ], 1)
        enc_input = patches + self.pos

        for enc_block in self.encoder:
            enc_input = enc_block(enc_input)

        linear_input = enc_input[:,[0,1]]
        output = self.linear(linear_input)

        return output


    def loss(self, inputs, class_targets):
        class_preds, teacher_preds = self(inputs)
        teacher_targets = self.teacher_model(inputs)
        return (self.MSE(class_preds  , class_targets) +
                self.MSE(teacher_preds, teacher_targets))

