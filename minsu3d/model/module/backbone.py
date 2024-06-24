import functools

import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
import torch.nn as nn
from minsu3d.model.module.common import ResidualBlock, UBlock

import MinkowskiEngine as ME

from .pointnext import PointNeXt


class Backbone(pl.LightningModule):
    def __init__(self, input_channel, output_channel, block_channels, block_reps, sem_classes):
        super().__init__()

        # # 1. U-Net
        # self.unet = nn.Sequential(
        #     ME.MinkowskiConvolution(in_channels=input_channel, out_channels=output_channel, kernel_size=3, dimension=3),
        #     UBlock([output_channel * c for c in block_channels], ME.MinkowskiBatchNorm, block_reps, ResidualBlock),
        #     ME.MinkowskiBatchNorm(output_channel),
        #     ME.MinkowskiReLU(inplace=True)
        # )

        # 1. PointNeXt
        self.pointnext = PointNeXt()  # add your pointnext module here

        # 2.1 semantic prediction branch
        self.semantic_branch = nn.Sequential(
            nn.Linear(output_channel, output_channel),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel, sem_classes)
        )

        # 2.2 offset prediction branch
        self.offset_branch = nn.Sequential(
            nn.Linear(output_channel, output_channel),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel, 3)
        )

    def forward(self, input_dict):
        output_dict = {}
        point_features_dense = self.pointnext(input_dict)  # here the input size is [batch_size, point_num, feature_dims]
        # convert the dense tensor to a sparse one, now the shape is [batch_sizexpoint_num, feature_dims]
        batch_size, feature_dims, point_num = point_features_dense.shape
        point_features_sparse = torch.transpose(torch.flatten(torch.transpose(point_features_dense, 0, 1), start_dim=1), 0, 1)
        output_dict["point_features"] = point_features_sparse
        output_dict["semantic_scores"] = self.semantic_branch(output_dict["point_features"])
        output_dict["point_offsets"] = self.offset_branch(output_dict["point_features"])
        return output_dict

