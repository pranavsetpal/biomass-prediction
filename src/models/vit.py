import numpy as np
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, n_patches, n_features, n_heads, dtype=torch.float64):
        super().__init__()

        self.layer_norm = nn.LayerNorm((n_patches, n_features), elementwise_affine=False, dtype=dtype)
        self.msa = nn.MultiheadAttention(n_features, n_heads, batch_first=True, dtype=dtype)
        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_features, dtype=dtype),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(n_features, n_features, dtype=dtype),
            nn.Dropout()
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

        self.learnable_emb = nn.Parameter(torch.zeros(1, n_features))
        self.pos = nn.Parameter(self.pos_emb(n_patches+1, n_features))

        self.encoder = nn.ModuleList([
            Encoder(n_patches+1, n_features, patch_size, dtype)
            for _ in range(n_enc_blocks)
        ])

        self.linear = nn.Linear(n_features, 1, dtype=dtype)

        self.MSE = nn.MSELoss(reduction='none')


    def forward(self, input):
        patches = self.patched(input, self.patch_size)

        patches = torch.cat([self.learnable_emb.unsqueeze(0).repeat(len(patches),1,1), patches], 1)
        enc_input = patches + self.pos

        for enc_block in self.encoder:
            enc_input = enc_block(enc_input)

        linear_input = enc_input[:,0]
        output = self.linear(linear_input)

        return output


    def fit(self, inputs, targets, epochs, batch_size, lr=1e-2):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        n_samples = len(inputs)
        inputs  = [ inputs[i:i+batch_size] for i in range(0,len(inputs) ,batch_size)]
        targets = [targets[i:i+batch_size] for i in range(0,len(targets),batch_size)]

        for epoch in range(epochs):
            total_loss = 0

            for input,target in zip(inputs,targets):
                optimizer.zero_grad()
                loss = self.MSE(self(input), target)
                total_loss += loss.sum().item()

                loss.backward(torch.ones(batch_size).unsqueeze(1))
                optimizer.step()

            total_loss /= n_samples

            if (total_loss < 100):
                print("---------------------")
                print("Num of epoches:", epoch)
                print("\tMSE  = ", total_loss)
                print("\tRMSE = ", total_loss**(1/2))
                print("---------------------")
                break

            print(f"Epoch {epoch+1}/{epochs}: \tMSE  = {total_loss:.4f} \tRSME = {total_loss**(1/2):.4f}")


    def eval(self, inputs, targets):
        loss = self.MSE(self(inputs), targets).sum().item()

        print("Eval:")
        print(f"MSE  = {loss:.4f}")
        print(f"RMSE = {loss**(1/2):.4f}")


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
