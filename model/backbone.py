from model.common import ResidualBlock, UBlock
import pytorch_lightning as pl
import MinkowskiEngine as ME
import torch.nn as nn
import functools


class Backbone(pl.LightningModule):
    def __init__(self, input_channel, output_chanel, block_channels, block_reps, sem_classes):
        super().__init__()

        sp_norm = functools.partial(ME.MinkowskiBatchNorm, eps=1e-4, momentum=0.1)
        norm = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # 1. U-Net
        self.unet = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=input_channel, out_channels=output_chanel, kernel_size=3, dimension=3),
            UBlock([output_chanel * c for c in block_channels], sp_norm, block_reps, ResidualBlock),
            sp_norm(output_chanel),
            ME.MinkowskiReLU(inplace=True)
        )

        # 2.1 semantic prediction branch
        self.semantic_branch = nn.Sequential(
            nn.Linear(output_chanel, output_chanel),
            norm(output_chanel),
            nn.ReLU(inplace=True),
            nn.Linear(output_chanel, sem_classes)
        )

        # 2.2 offset prediction branch
        self.offset_branch = nn.Sequential(
            nn.Linear(output_chanel, output_chanel),
            norm(output_chanel),
            nn.ReLU(inplace=True),
            nn.Linear(output_chanel, 3)
        )

    def forward(self, voxel_features, voxel_coordinates, v2p_map):
        output_dict = {}
        x = ME.SparseTensor(features=voxel_features, coordinates=voxel_coordinates.int())
        unet_out = self.unet(x)
        point_features = unet_out.features[v2p_map.long()]
        semantic_scores = self.semantic_branch(point_features)
        point_offsets = self.offset_branch(point_features)
        output_dict["semantic_scores"] = semantic_scores
        output_dict["point_offsets"] = point_offsets
        return output_dict
