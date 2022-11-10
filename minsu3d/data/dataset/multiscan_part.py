import os
from tqdm import tqdm
import numpy as np
import torch
from minsu3d.data.dataset import GeneralDataset


class MultiScanPart(GeneralDataset):
    def __int__(self, cfg, split):
        super().__init__(cfg, split)
