from model.module.common import ResidualBlock, UBlock
import pytorch_lightning as pl
import MinkowskiEngine as ME
import torch.nn as nn
import functools


class Backbone(pl.LightningModule):
    def __init__(self, input_channel, output_channel, block_channels, block_reps, sem_classes):
        super().__init__()

        sp_norm = functools.partial(ME.MinkowskiBatchNorm)
        norm = functools.partial(nn.BatchNorm1d)

        # 1. U-Net
        self.unet = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=input_channel, out_channels=output_channel, kernel_size=3, dimension=3),
            UBlock([output_channel * c for c in block_channels], sp_norm, block_reps, ResidualBlock),
            sp_norm(output_channel),
            ME.MinkowskiReLU(inplace=True)
        )

        # 2.1 semantic prediction branch
        self.semantic_branch = nn.Sequential(
            nn.Linear(output_channel, output_channel),
            norm(output_channel),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel, sem_classes)
        )

        # 2.2 offset prediction branch
        self.offset_branch = nn.Sequential(
            nn.Linear(output_channel, output_channel),
            norm(output_channel),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel, 3)
        )

    def forward(self, voxel_features, voxel_coordinates, v2p_map):
        output_dict = {}
        x = ME.SparseTensor(features=voxel_features, coordinates=voxel_coordinates.int())
        unet_out = self.unet(x)
        output_dict["point_features"] = unet_out.features[v2p_map.long()]
        output_dict["semantic_scores"] = self.semantic_branch(output_dict["point_features"])
        output_dict["point_offsets"] = self.offset_branch(output_dict["point_features"])
        return output_dict
