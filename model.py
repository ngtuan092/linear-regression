import torch
import torch.nn as nn
import torch.nn.functional as F


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







