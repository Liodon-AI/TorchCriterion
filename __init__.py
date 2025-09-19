from torchcriterion.classification.cross_entropy import CrossEntropyLoss
from torchcriterion.classification.focal import FocalLoss
from torchcriterion.regression.mse import MSELoss
from torchcriterion.regression.huber import HuberLoss
from torchcriterion.segmentation.dice import DiceLoss
from torchcriterion.segmentation.tversky import TverskyLoss
from torchcriterion.metric_learning.triplet import TripletLoss
from torchcriterion.metric_learning.contrastive import ContrastiveLoss

__all__ = [
    "CrossEntropyLoss",
    "FocalLoss",
    "MSELoss",
    "HuberLoss",
    "DiceLoss",
    "TverskyLoss",
    "TripletLoss",
    "ContrastiveLoss",
]
