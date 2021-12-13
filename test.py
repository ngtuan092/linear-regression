import torch
from model import LinearModel
from dataLoader import val_ds

def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(input)               # fill this
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)


if __name__ == '__main__':
    model = LinearModel(6, 1)
    model.load_state_dict(torch.load('InsuranceModel'))
    inputs, targets = val_ds[3]
    predict_single(inputs, targets, model)