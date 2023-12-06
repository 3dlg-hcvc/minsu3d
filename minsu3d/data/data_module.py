import torch
import MinkowskiEngine as ME
import pytorch_lightning as pl
from importlib import import_module
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


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

    vert_batch_ids = []

    instance_center_xyz = []
    instance_num_point = []
    sem_labels = []
    instance_ids = []
    total_num_inst = 0
    batch_data = []

    default_collate_items = ("scan_id", "point_xyz", "point_rgb", "point_normal",)
    scan_ids = []

    for i, b in enumerate(batch):
        scan_ids.append(b["scan_id"])
        batch_data.append({k: b[k] for k in default_collate_items})
        sem_labels.append(torch.from_numpy(b["sem_labels"]))
        vert_batch_ids.append(torch.full((b["point_xyz"].shape[0],), fill_value=i, dtype=torch.uint8))

        instance_ids_i = b["instance_ids"]
        instance_ids_i[instance_ids_i != -1] += total_num_inst
        total_num_inst += b["num_instance"].item()
        instance_ids.append(torch.from_numpy(instance_ids_i))

        instance_center_xyz.append(torch.from_numpy(b["instance_center_xyz"]))
        instance_num_point.append(torch.from_numpy(b["instance_num_point"]))

    data = default_collate(batch_data)  # default collate_fn

    data["vert_batch_ids"] = torch.cat(vert_batch_ids, dim=0)
    data["sem_labels"] = torch.cat(sem_labels, dim=0)
    data["instance_ids"] = torch.cat(instance_ids, dim=0)
    data["instance_center_xyz"] = torch.cat(instance_center_xyz, dim=0)
    data["instance_num_point"] = torch.cat(instance_num_point, dim=0)

    data['scan_ids'] = scan_ids

    return data
