import copy

import hydra
import torch
import torch.nn as nn

from .openpoints.models import build_model_from_cfg
from .openpoints.utils import EasyConfig


class PointNeXt(nn.Module):
    def __init__(self,
                 cfg="./minsu3d/model/module/pointnext/cfgs/partnetsim/pointnext-s.yaml"):
        super().__init__()
        self.cfg = EasyConfig()
        self.cfg.load(cfg, recursive=True)
        #cfg.update(opts)  # overwrite the default arguments in yml
        encoder_cfg = self.cfg.model.encoder_args
        decoder_cfg = self.cfg.model.decoder_args

        self.encoder = build_model_from_cfg(encoder_cfg)
        if decoder_cfg is not None:
            decoder_args_merged_with_encoder = copy.deepcopy(encoder_cfg)
            decoder_args_merged_with_encoder.update(decoder_cfg)
            decoder_args_merged_with_encoder.encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder,
                                                                                                         'channel_list') else None
            self.decoder = build_model_from_cfg(decoder_args_merged_with_encoder)
        else:
            self.decoder = None

    def forward(self, data):
        p, f = self.encoder.forward_seg_feat(data)
        if self.decoder is not None:
            f = self.decoder(p, f).squeeze(-1)
        return f
