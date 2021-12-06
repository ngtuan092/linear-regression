import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.optim import SGD
from utils import DeviceDataLoader, to_device, get_device

from torch.utils.data import DataLoader, random_split, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets.utils import download_url


# download dataset from url and save into this file

# convert into tensor


def dataframe_to_arrays(dataframe):
    dataframe1 = dataframe.copy(deep=True)
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    inputs = dataframe1[[col for col in dataframe1.columns[:-1]]].to_numpy()
    targets = dataframe1[[dataframe1.columns[-1]]].to_numpy()
    return inputs, targets


def split(dataset, val_percent):
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_percent)
    train_size = dataset_size - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return train_ds, val_ds


# batch size


class LinearModel(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()
        self.linear = nn.Linear(num_input, num_output)

    def forward(self, xb):
        out = self.linear(xb)
        return out

    def computeLoss(self, batch):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.mse_loss(outputs, targets)
        return loss

    def batch_evaluate(self, val_batch):
        inputs, targets = val_batch
        outputs = self(inputs)
        loss = F.mse_loss(outputs, targets)
        return {'val_loss': loss.detach()}

    def epoch_evaluate(self, batch_results):
        batch_losses = [x['val_loss'] for x in batch_results]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 1000 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(
                epoch+1, result['val_loss']))


def evaluate(model, val_loader):
    outputs = [model.batch_evaluate(batch) for batch in val_loader]
    return model.epoch_evaluate(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = []
    for epoch in range(epochs):
        for batch in train_loader:
            loss = model.computeLoss(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        results = evaluate(model, val_loader)
        model.epoch_end(epoch, results, epochs)
        history.append(results)
    return history


if __name__ == '__main__':
    DATASET_URL = "https://hub.jovian.ml/wp-content/uploads/2020/05/insurance.csv"
    DATA_FILENAME = "insurance.csv"
    device = get_device()
    download_url(DATASET_URL, '.')
    dataframe_raw = pd.read_csv(DATA_FILENAME)
    # list of categorical columns
    categorical_cols = [col_name for col_name in dataframe_raw.select_dtypes(exclude=[
        "number"])]
    inputs, targets = dataframe_to_arrays(dataframe_raw)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    dataset = TensorDataset(inputs, targets)
    train_ds, val_ds = split(dataset, 0.2)
    NUM_BATCH = 128
    train_loader = DataLoader(
        train_ds, batch_size=NUM_BATCH, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=NUM_BATCH,
                            num_workers=4, pin_memory=True)
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    InsuranceModel = LinearModel(6, 1)
    to_device(InsuranceModel, device)
    fit(10000, 1e-4, InsuranceModel, train_loader, val_loader)
