import pytorch_lightning as pl
import torch.nn as nn


class ClassificationLoss(pl.LightningModule):

    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predictions, gts):
        return self.criterion(predictions, gts)