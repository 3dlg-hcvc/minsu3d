from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from minsu3d.common_ops.functions import common_ops


class DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.dataset = getattr(import_module('minsu3d.data.dataset'), data_cfg.data.dataset)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = self.dataset(self.data_cfg, "train")
            self.val_set = self.dataset(self.data_cfg, "val")
        if stage == "test" or stage is None:
            self.val_set = self.dataset(self.data_cfg, self.data_cfg.model.inference.split)
        if stage == "predict" or stage is None:
            self.test_set = self.dataset(self.data_cfg, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.data_cfg.data.batch_size, shuffle=True, pin_memory=True,
                          collate_fn=sparse_collate_fn, num_workers=self.data_cfg.data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, pin_memory=True, collate_fn=sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, pin_memory=True, collate_fn=sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, pin_memory=True, collate_fn=sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)


def sparse_collate_fn(batch):
    data = {}
    locs = []
    locs_scaled = []
    vert_batch_ids = []
    feats = []
    sem_labels = []
    instance_ids = []
    instance_info = []
    instance_num_point = []
    instance_offsets = [0]
    total_num_inst = 0
    instance_cls = []

    scan_ids = []

    for i, b in enumerate(batch):
        scan_ids.append(b["scan_id"])
        locs.append(torch.from_numpy(b["locs"]))

        locs_scaled.append(torch.from_numpy(b["locs_scaled"]).int())
        vert_batch_ids.append(torch.full((b["locs_scaled"].shape[0],), fill_value=i, dtype=torch.int16))
        feats.append(torch.from_numpy(b["feats"]))

        instance_ids_i = b["instance_ids"]
        instance_ids_i[instance_ids_i != -1] += total_num_inst
        total_num_inst += b["num_instance"].item()
        instance_ids.append(torch.from_numpy(instance_ids_i))

        sem_labels.append(torch.from_numpy(b["sem_labels"]))

        instance_info.append(torch.from_numpy(b["instance_info"]))
        instance_num_point.append(torch.from_numpy(b["instance_num_point"]))
        instance_offsets.append(instance_offsets[-1] + b["num_instance"].item())

        instance_cls.extend(b["instance_semantic_cls"])

    tmp_locs_scaled = torch.cat(locs_scaled, dim=0)
    data['scan_ids'] = scan_ids
    data["locs"] = torch.cat(locs, dim=0)
    data["vert_batch_ids"] = torch.cat(vert_batch_ids, dim=0)
    data["feats"] = torch.cat(feats, dim=0)

    data["sem_labels"] = torch.cat(sem_labels, dim=0)
    data["instance_ids"] = torch.cat(instance_ids, dim=0)
    data["instance_info"] = torch.cat(instance_info, dim=0)
    data["instance_num_point"] = torch.cat(instance_num_point, dim=0)
    data["instance_offsets"] = torch.tensor(instance_offsets, dtype=torch.int32)
    data["instance_semantic_cls"] = torch.tensor(instance_cls, dtype=torch.int32)

    # voxelize
    data["voxel_locs"], data["v2p_map"], data["p2v_map"] = common_ops.voxelization_idx(tmp_locs_scaled,
                                                                                       data["vert_batch_ids"],
                                                                                       len(batch),
                                                                                       4)
    return data
