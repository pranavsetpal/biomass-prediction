import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Lasso(nn.Module):
  def __init__(self, image_size, n_channels, alpha, dtype=torch.float64):
    super().__init__()
    n_features = image_size**2 * n_channels

    self.alpha = alpha
    self.weights = nn.Parameter(torch.randn(n_features, dtype=dtype))
    self.biases = nn.Parameter(torch.zeros(1))

    self.MSE = nn.MSELoss(reduction="sum")


  def forward(self, inputs):
    norm_inputs = (inputs-inputs.mean()) / inputs.std()
    flattened_inputs = norm_inputs.reshape(len(norm_inputs), -1)

    weights = self.weights.unsqueeze(0).repeat(len(inputs),1)
    biases = self.biases.unsqueeze(0).repeat(len(inputs),1)
    output = torch.bmm(flattened_inputs.unsqueeze(1), weights.unsqueeze(2)).squeeze(2) + biases

    return output


  def loss(self, inputs, targets):
    preds = self(inputs)
    return self.MSE(preds, targets) + self.alpha*(self.weights).abs().sum()

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
    return val_loss
