import functools
import torch.nn as nn
import pytorch_lightning as pl
import MinkowskiEngine as ME
from min3d.model.module.common import ResidualBlock, UBlock


class TinyUnet(pl.LightningModule):
    def __init__(self, channel):
        super().__init__()

        sp_norm = functools.partial(ME.MinkowskiBatchNorm)

        # 1. U-Net
        self.unet = nn.Sequential(
            UBlock([channel, 2 * channel], sp_norm, 2, ResidualBlock),
            sp_norm(channel),
            ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, proposals_voxel_feats):
        return self.unet(proposals_voxel_feats)
