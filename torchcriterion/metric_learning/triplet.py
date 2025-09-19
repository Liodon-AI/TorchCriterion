import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)
