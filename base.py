import torch.nn as nn

class BaseCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        raise NotImplementedError("Each criterion must implement the forward method.")
