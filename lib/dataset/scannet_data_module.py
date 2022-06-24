import pytorch_lightning as pl
import tqdm
import torch
import os


class ScanNetDataModule(pl.LightningDataModule):
    def __int__(self, cfg):
        super().__init__()
        self.DATA_MAP = {
            "train": cfg.SCANNETV2_PATH.train_list,
            "val": cfg.SCANNETV2_PATH.val_list,
            "test": cfg.SCANNETV2_PATH.test_list
        }

    def setup(self, stage=None):
        # read preprocessed pth data from disk
        self.scenes = [
            torch.load(os.path.join(self.root, self.split, d + self.file_suffix))
            for d in tqdm(self.scene_names)
        ]

        if stage == "fit" or stage is None:
            self.scannet_train = 1
