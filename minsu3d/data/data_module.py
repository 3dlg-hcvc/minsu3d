import torch
import MinkowskiEngine as ME
import pytorch_lightning as pl
from importlib import import_module
from torch.utils.data import DataLoader


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
        return DataLoader(
            self.train_set, batch_size=self.data_cfg.data.batch_size, shuffle=True,
            pin_memory=True, collate_fn=_sparse_collate_fn, num_workers=self.data_cfg.data.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=1, pin_memory=True,
            collate_fn=_sparse_collate_fn, num_workers=self.data_cfg.data.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=1, pin_memory=True,
            collate_fn=_sparse_collate_fn, num_workers=self.data_cfg.data.num_workers
        )


def _sparse_collate_fn(batch):
    data = {}
    point_xyz = []
    vert_batch_ids = []
    sem_labels = []
    instance_ids = []
    instance_center_xyz = []
    instance_num_point = []
    instance_offsets = [0]
    total_num_inst = 0
    instance_cls = []
    voxel_xyz_list = []
    voxel_features_list = []
    voxel_point_map_list = []
    num_voxel_batch = 0
    scan_ids = []

    for i, b in enumerate(batch):
        scan_ids.append(b["scan_id"])
        point_xyz.append(torch.from_numpy(b["point_xyz"]))

        voxel_xyz_list.append(b["voxel_xyz"])
        voxel_features_list.append(b["voxel_features"])
        voxel_point_map_list.append(b["voxel_point_map"] + num_voxel_batch)
        num_voxel_batch += b["voxel_xyz"].shape[0]

        vert_batch_ids.append(torch.full((b["point_xyz"].shape[0],), fill_value=i, dtype=torch.uint8))

        instance_ids_i = b["instance_ids"]
        instance_ids_i[instance_ids_i != -1] += total_num_inst
        total_num_inst += b["num_instance"].item()
        instance_ids.append(torch.from_numpy(instance_ids_i))

        sem_labels.append(torch.from_numpy(b["sem_labels"]))

        instance_center_xyz.append(torch.from_numpy(b["instance_center_xyz"]))
        instance_num_point.append(torch.from_numpy(b["instance_num_point"]))
        instance_offsets.append(instance_offsets[-1] + b["num_instance"].item())

        instance_cls.extend(b["instance_semantic_cls"])

    data['scan_ids'] = scan_ids
    data["point_xyz"] = torch.cat(point_xyz, dim=0)
    data["vert_batch_ids"] = torch.cat(vert_batch_ids, dim=0)

    data["sem_labels"] = torch.cat(sem_labels, dim=0)
    data["instance_ids"] = torch.cat(instance_ids, dim=0)
    data["instance_center_xyz"] = torch.cat(instance_center_xyz, dim=0)
    data["instance_num_point"] = torch.cat(instance_num_point, dim=0)
    data["instance_offsets"] = torch.tensor(instance_offsets, dtype=torch.int32)
    data["instance_semantic_cls"] = torch.tensor(instance_cls, dtype=torch.int16)

    data["voxel_xyz"], data["voxel_features"] = ME.utils.sparse_collate(
        coords=voxel_xyz_list, feats=voxel_features_list
    )
    data["voxel_point_map"] = torch.cat(voxel_point_map_list, dim=0)
    return data
