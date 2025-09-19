import torch.nn as nn


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.loss = nn.SmoothL1Loss(beta=delta)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)
