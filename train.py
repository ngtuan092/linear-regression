import torch.nn.functional as F
import torch
import torch.nn.functional as F
from dataLoader import train_loader, val_loader 
from model import LinearModel 
# download dataset from url and save into this file

# convert into tensor
def evaluate(model, val_loader):
    outputs = [model.batch_evaluate(batch) for batch in val_loader]
    return model.epoch_evaluate(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
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
    model = LinearModel(6, 1)
    num_epochs = 10000
    lr = 1e-4
    history = fit(num_epochs, lr, model, train_loader, val_loader)
    