import torch.nn as nn


class ClassificationLoss(nn.Module):

    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predictions, gts):
        return self.criterion(predictions, gts)