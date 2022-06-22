import os
import sys
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from MinkowskiEngine.utils import sparse_collate, batched_coordinates

sys.path.append("../")  # HACK add the lib folder
from lib.softgroup_ops.functions import softgroup_ops
from lib.utils.pc import crop
from lib.utils.transform import jitter, flip, rotz, elastic


class ScanNet(Dataset):

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.root = cfg.SCANNETV2_PATH.splited_data
        self.file_suffix = cfg.data.file_suffix

        self.full_scale = cfg.data.full_scale
        self.scale = cfg.data.scale
        self.max_num_point = cfg.data.max_num_point
        self.mode = cfg.data.mode

        self.requires_gt_mask = cfg.data.requires_gt_mask

        self.DATA_MAP = {
            "train": cfg.SCANNETV2_PATH.train_list,
            "val": cfg.SCANNETV2_PATH.val_list,
            "test": cfg.SCANNETV2_PATH.test_list
        }

        self.multiview_data = {}

        self._load()

    def _load(self):
        with open(self.DATA_MAP[self.split]) as f:
            self.scene_names = [l.rstrip() for l in f]

        if self.cfg.data.mini:
            self.scene_names = self.scene_names[:self.cfg.data.mini]

        self.scenes = [
            torch.load(os.path.join(self.root, self.split, d + self.file_suffix))
            for d in tqdm(self.scene_names)
        ]

        for scene_data in self.scenes:
            mesh = scene_data["aligned_mesh"]
            mesh[:, :3] -= mesh[:, :3].mean(0)
            mesh[:, 3:6] = mesh[:, 3:6] / 127.5 - 1

    def __len__(self):
        return len(self.scenes)

    def _augment(self, xyz, return_mat=False):
        m = np.eye(3)
        if self.cfg.data.augmentation.jitter_xyz:
            m = np.matmul(m, jitter())
        if self.cfg.data.augmentation.flip:
            flip_m = flip(0, random=True)
            m *= flip_m
        if self.cfg.data.augmentation.rotation:
            t = np.random.rand() * 2 * np.pi
            rot_m = rotz(t)
            m = np.matmul(m, rot_m)  # rotation around z
        if return_mat:
            return np.matmul(xyz, m), m
        else:
            return np.matmul(xyz, m)

    def _croppedInstanceIds(self, instance_ids, valid_idxs):
        """
        Postprocess instance_ids after cropping
        """
        instance_ids = instance_ids[valid_idxs]
        j = 0
        while (j < instance_ids.max()):
            if (len(np.where(instance_ids == j)[0]) == 0):
                instance_ids[instance_ids == instance_ids.max()] = j
            j += 1
        return instance_ids

    def _getInstanceInfo(self, xyz, instance_ids, sem_labels=None):
        """
        :param xyz: (n, 3)
        :param instance_ids: (n), int, (0~nInst-1, -1)
        :return: num_instance, dict
        """
        instance_num_point = []  # (nInst), int
        unique_instance_ids = np.unique(instance_ids)
        num_instance = len(unique_instance_ids) - 1 if -1 in unique_instance_ids else len(unique_instance_ids)
        instance_info = np.zeros(
            (xyz.shape[0], 12), dtype=np.float32
        )  # (n, 12), float, (meanx, meany, meanz, cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_cls = []
        for k, i_ in enumerate(unique_instance_ids, -1):
            if i_ < 0: continue

            inst_i_idx = np.where(instance_ids == i_)

            # instance_info
            xyz_i = xyz[inst_i_idx]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            c_xyz_i = (max_xyz_i + min_xyz_i) / 2
            instance_info_i = instance_info[inst_i_idx]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = c_xyz_i
            instance_info_i[:, 6:9] = min_xyz_i
            instance_info_i[:, 9:12] = max_xyz_i
            instance_info[inst_i_idx] = instance_info_i

            # instance_num_point
            instance_num_point.append(inst_i_idx[0].size)

            # semantic label
            cls_idx = inst_i_idx[0][0]
            instance_cls.append(sem_labels[cls_idx] - len(self.cfg.data.ignore_classes) if sem_labels[cls_idx] != self.cfg.data.ignore_label else sem_labels[cls_idx])

        return num_instance, instance_info, instance_num_point, instance_cls

    def _generate_gt_clusters(self, points, instance_ids):
        gt_proposals_idx = []
        gt_proposals_offset = [0]
        unique_instance_ids = np.unique(instance_ids)
        num_instance = len(unique_instance_ids) - 1 if -1 in unique_instance_ids else len(unique_instance_ids)
        instance_bboxes = np.zeros((num_instance, 6))

        object_ids = []
        for cid, i_ in enumerate(unique_instance_ids, -1):
            if i_ < 0:
                continue
            object_ids.append(i_)
            inst_i_idx = np.where(instance_ids == i_)[0]
            inst_i_points = points[inst_i_idx]
            xmin = np.min(inst_i_points[:, 0])
            ymin = np.min(inst_i_points[:, 1])
            zmin = np.min(inst_i_points[:, 2])
            xmax = np.max(inst_i_points[:, 0])
            ymax = np.max(inst_i_points[:, 1])
            zmax = np.max(inst_i_points[:, 2])
            bbox = np.array(
                [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, xmax - xmin, ymax - ymin, zmax - zmin])
            instance_bboxes[cid, :] = bbox

            proposals_idx_i = np.vstack((np.ones(len(inst_i_idx)) * cid, inst_i_idx)).transpose().astype(np.int32)
            gt_proposals_idx.append(proposals_idx_i)
            gt_proposals_offset.append(len(inst_i_idx) + gt_proposals_offset[-1])

        gt_proposals_idx = np.concatenate(gt_proposals_idx, axis=0).astype(np.int32)
        gt_proposals_offset = np.array(gt_proposals_offset).astype(np.int32)

        return gt_proposals_idx, gt_proposals_offset, object_ids, instance_bboxes

    def __getitem__(self, id):
        scene_id = self.scene_names[id]
        scene = self.scenes[id]

        mesh = scene["aligned_mesh"]

        points = mesh[:, :3]  # (N, 3)
        feats = mesh[:, 3:6]  # (N, 3) rgb

        data = {"id": id, "scene_id": scene_id}

        if self.split != "test":
            instance_ids = scene["instance_ids"]
            sem_labels = scene["sem_labels"]  # {0,1,...,19}, -1 as ignored (unannotated) class

            # augment
            if self.split == "train":
                points_augment = self._augment(points)
            else:
                points_augment = points.copy()

            # scale
            points = points_augment * self.scale

            # elastic
            if self.split == "train" and self.cfg.data.augmentation.elastic:
                points = elastic(points, 6 * self.scale // 50, 40 * self.scale / 50)
                points = elastic(points, 20 * self.scale // 50, 160 * self.scale / 50)

            # jitter rgb
            if self.split == "train" and self.cfg.data.augmentation.jitter_rgb:
                feats[0:3] += np.random.randn(3) * 0.1


            # offset
            points -= points.min(0)

            if self.split == "train":
                # crop
                points, valid_idxs = crop(points, self.max_num_point, self.full_scale[1])
                # points, valid_idxs = random_sampling(points, self.max_num_point, return_choices=True)

                points = points[valid_idxs]
                points_augment = points_augment[valid_idxs]
                feats = feats[valid_idxs]
                sem_labels = sem_labels[valid_idxs]
                instance_ids = self._croppedInstanceIds(instance_ids, valid_idxs)

            num_instance, instance_info, instance_num_point, instance_semantic_cls = self._getInstanceInfo(
                points_augment, instance_ids.astype(np.int32), sem_labels)

            if self.requires_gt_mask:
                gt_proposals_idx, gt_proposals_offset, _, _ = self._generate_gt_clusters(points, instance_ids)

            data["locs"] = points_augment.astype(np.float32)  # (N, 3)
            data["locs_scaled"] = points.astype(np.float32)  # (N, 3)
            data["feats"] = feats.astype(np.float32)  # (N, 3)
            data["sem_labels"] = sem_labels.astype(np.int32)  # (N,)
            data["instance_ids"] = instance_ids.astype(np.int32)  # (N,) 0~total_nInst, -1
            data["num_instance"] = np.array(num_instance, dtype=np.int32)  # int
            data["instance_info"] = instance_info.astype(np.float32)  # (N, 12)
            data["instance_num_point"] = np.array(instance_num_point, dtype=np.int32)  # (num_instance,)
            data["instance_semantic_cls"] = np.array(instance_semantic_cls, dtype=np.int32)
            if self.requires_gt_mask:
                data['gt_proposals_idx'] = gt_proposals_idx
                data['gt_proposals_offset'] = gt_proposals_offset
        else:
            # scale
            points = points.copy() * self.scale

            # offset
            points -= points.min(0)

            data["locs"] = points.astype(np.float32)  # (N, 3)
            data["locs_scaled"] = points.astype(np.float32)  # (N, 3)
            data["feats"] = feats.astype(np.float32)  # (N, 3)

        return data


def scannet_loader(cfg):
    def scannet_collate_fn(batch):
        batch_size = batch.__len__()
        data = {}
        for key in batch[0].keys():
            if key in ['locs', 'locs_scaled', 'feats', 'sem_labels', 'instance_ids', 'num_instance', 'instance_info',
                       'instance_num_point', "instance_semantic_cls", 'gt_proposals_idx', 'gt_proposals_offset']:
                continue
            if isinstance(batch[0][key], tuple):
                coords, feats = list(zip(*[sample[key] for sample in batch]))
                coords_b = batched_coordinates(coords)
                feats_b = torch.from_numpy(np.concatenate(feats, 0)).float()
                data[key] = (coords_b, feats_b)
            elif isinstance(batch[0][key], np.ndarray):
                data[key] = torch.stack(
                    [torch.from_numpy(sample[key]) for sample in batch])
            elif isinstance(batch[0][key], torch.Tensor):
                data[key] = torch.stack([sample[key] for sample in batch])
            elif isinstance(batch[0][key], dict):
                data[key] = sparse_collate([sample[key] for sample in batch])
            else:
                data[key] = [sample[key] for sample in batch]
        return data

    def sparse_collate_fn(batch):
        data = scannet_collate_fn(batch)

        locs = []
        locs_scaled = []
        feats = []
        sem_labels = []
        instance_ids = []
        instance_info = []  # (N, 12)
        instance_num_point = []  # (total_nInst), int
        batch_offsets = [0]
        instance_offsets = [0]
        total_num_inst = 0
        total_points = 0
        instance_cls = []  # (total_nInst), long
        gt_proposals_idx = []
        gt_proposals_offset = []

        for i, b in enumerate(batch):
            locs.append(torch.from_numpy(b["locs"]))
            locs_scaled.append(
                torch.cat([
                    torch.LongTensor(b["locs_scaled"].shape[0], 1).fill_(i),
                    torch.from_numpy(b["locs_scaled"]).long()
                ], 1))

            feats.append(torch.from_numpy(b["feats"]))
            batch_offsets.append(batch_offsets[-1] + b["locs_scaled"].shape[0])

            if cfg.general.task != "test":
                if cfg.data.requires_gt_mask:
                    gt_proposals_idx_i = b["gt_proposals_idx"]
                    gt_proposals_idx_i[:, 0] += total_num_inst
                    gt_proposals_idx_i[:, 1] += total_points
                    gt_proposals_idx.append(torch.from_numpy(b["gt_proposals_idx"]))
                    if gt_proposals_offset != []:
                        gt_proposals_offset_i = b["gt_proposals_offset"]
                        gt_proposals_offset_i += gt_proposals_offset[-1][-1].item()
                        gt_proposals_offset.append(torch.from_numpy(gt_proposals_offset_i[1:]))
                    else:
                        gt_proposals_offset.append(torch.from_numpy(b["gt_proposals_offset"]))
                instance_ids_i = b["instance_ids"]
                instance_ids_i[np.where(instance_ids_i != -1)] += total_num_inst
                total_num_inst += b["num_instance"].item()
                total_points += len(instance_ids_i)
                instance_ids.append(torch.from_numpy(instance_ids_i))

                sem_labels.append(torch.from_numpy(b["sem_labels"]))

                instance_info.append(torch.from_numpy(b["instance_info"]))
                instance_num_point.append(torch.from_numpy(b["instance_num_point"]))
                instance_offsets.append(instance_offsets[-1] + b["num_instance"].item())

                instance_cls.extend(b["instance_semantic_cls"])

        data["locs"] = torch.cat(locs, 0)  # float (N, 3)
        data["locs_scaled"] = torch.cat(locs_scaled, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        data["feats"] = torch.cat(feats, 0)  # .to(torch.float32)            # float (N, C)
        data["batch_offsets"] = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        if cfg.general.task != "test":
            if cfg.data.requires_gt_mask:
                data["gt_proposals_idx"] = torch.cat(gt_proposals_idx, 0).to(torch.int32)
                data["gt_proposals_offset"] = torch.cat(gt_proposals_offset, 0).to(torch.int32)
            data["sem_labels"] = torch.cat(sem_labels, 0).long()  # long (N,)
            data["instance_ids"] = torch.cat(instance_ids, 0).long()  # long, (N,)
            data["instance_info"] = torch.cat(instance_info, 0)  # float (total_nInst, 12)
            data["instance_num_point"] = torch.cat(instance_num_point, 0)  # (total_nInst)
            data["instance_offsets"] = torch.tensor(instance_offsets, dtype=torch.int)  # int (B+1)
            data["instance_semantic_cls"] = torch.tensor(instance_cls, dtype=torch.long)  # long (total_nInst)
        ### voxelize
        data["voxel_locs"], data["p2v_map"], data["v2p_map"] = softgroup_ops.voxelization_idx(data["locs_scaled"],
                                                                                              len(batch),
                                                                                              4)  # mode=4 TODO: the naming p2v is wrong! should be v2p

        return data

    if cfg.general.task == "train":
        splits = ["train", "val"]
    else:
        splits = [cfg.data.split]

    datasets = {split: ScanNet(cfg, split) for split in splits}

    dataloaders = {
        split:
            DataLoader(datasets[split],
                       batch_size=cfg.data.batch_size,
                       shuffle=True if split == "train" else False,
                       pin_memory=True,
                       collate_fn=sparse_collate_fn)
        for split in splits
    }

    return datasets, dataloaders
