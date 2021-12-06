from torch.utils.data import DataLoader, random_split, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets.utils import download_url


DATASET_URL = "https://hub.jovian.ml/wp-content/uploads/2020/05/insurance.csv"
DATA_FILENAME = "insurance.csv"

download_url(DATASET_URL, '.') # download dataset from url and save into this file


dataframe_raw = pd.read_csv(DATA_FILENAME)

# list of categorical columns
categorical_cols = [col_name for col_name in dataframe_raw.select_dtypes(exclude=["number"])]
# convert into tensor
def dataframe_to_arrays(dataframe):
    dataframe1 = dataframe.copy(deep=True)
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    inputs = dataframe1[[col for col in dataframe1.columns[:-1]]].to_numpy()
    targets = dataframe1[[dataframe1.columns[-1]]].to_numpy()
    return inputs, targets


inputs, targets = dataframe_to_arrays(dataframe_raw)
inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)
dataset = TensorDataset(inputs, targets)

def split(dataset, val_percent):
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_percent)
    train_size = dataset_size - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return train_ds, val_ds


# batch size
train_ds, val_ds = split(dataset, 0.2)
NUM_BATCH = 128
train_loader = DataLoader(train_ds, batch_size=NUM_BATCH, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=NUM_BATCH, num_workers=4, pin_memory=True)

