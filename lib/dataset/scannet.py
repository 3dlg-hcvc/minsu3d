import os
import sys
import time
import json
import h5py 
import torch

import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from MinkowskiEngine.utils import sparse_collate, batched_coordinates

from lib.utils.bbox import get_3d_box_batch

sys.path.append("../")  # HACK add the lib folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.pointgroup_ops.functions import pointgroup_ops
from lib.utils.pc import crop, random_sampling
from lib.utils.transform import jitter, flip, rotz, elastic

MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

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

        self.use_color = cfg.model.use_color
        self.use_multiview = cfg.model.use_multiview
        self.use_normal = cfg.model.use_normal
        self.requires_bbox = cfg.data.requires_bbox
        
        self.DATA_MAP = {
            "train": cfg.SCANNETV2_PATH.train_list,
            "val": cfg.SCANNETV2_PATH.val_list,
            "test": cfg.SCANNETV2_PATH.test_list
        }

        self.multiview_data = {}
        
        self._load()

    def _load(self):
        if self.requires_bbox:
            self.DC = ScannetDatasetConfig(self.cfg)
        
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
            mesh[:, 3:6] = (mesh[:, 3:6] - MEAN_COLOR_RGB) / 256.0
            # mesh[:, 3:6] = mesh[:, 3:6] / 127.5 - 1

    def __len__(self):
        return len(self.scenes)

    def _augment(self, xyz):
        m = np.eye(3)
        if self.cfg.data.transform.jitter:
            m *= jitter()
        if self.cfg.data.transform.flip:
            m *= flip(0, random=True)
        if self.cfg.data.transform.rot:
            t = np.random.rand() * 2 * np.pi
            m = np.matmul(m, rotz(t))  # rotation around z
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
        
        if self.requires_bbox:
            assert sem_labels is not None, "sem_labels are not provided"
            max_num_instance = self.cfg.data.max_num_instance # NOTE this should always be the same number
            instance_bboxes = np.zeros((max_num_instance, 6))
            instance_bboxes_semcls = np.zeros((max_num_instance))
            instance_bbox_ids = np.zeros((max_num_instance))
            angle_classes = np.zeros((max_num_instance,))
            angle_residuals = np.zeros((max_num_instance,))
            size_classes = np.zeros((max_num_instance,))
            size_residuals = np.zeros((max_num_instance, 3))
            bbox_label = np.zeros((max_num_instance))

        for k, i_ in enumerate(unique_instance_ids, -1):
            if i_ < 0: continue
            
            inst_i_idx = np.where(instance_ids == i_)

            ### instance_info
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

            ### instance_num_point
            instance_num_point.append(inst_i_idx[0].size)
            
            if self.requires_bbox:
                if k >= 128: continue
                instance_bboxes[k, :3] = c_xyz_i
                instance_bboxes[k, 3:] = max_xyz_i - min_xyz_i
                sem_cls = sem_labels[inst_i_idx][0]
                sem_cls = sem_cls - 2 if sem_cls >=  2 else 17
                instance_bboxes_semcls[k] = sem_cls
                instance_bbox_ids[k] = i_
                size_classes[k] = sem_cls
                size_residuals[k, :] = instance_bboxes[k, 3:] - self.DC.mean_size_arr[int(sem_cls),:]
                bbox_label[k] = 1
                
        if self.requires_bbox:
            return num_instance, instance_info, instance_num_point, instance_bboxes, instance_bboxes_semcls, instance_bbox_ids, angle_classes, angle_residuals, size_classes, size_residuals, bbox_label
        else:
            return num_instance, instance_info, instance_num_point

    def __getitem__(self, id):
        scene_id = self.scene_names[id]
        scene = self.scenes[id]

        mesh = scene["aligned_mesh"]

        data = mesh[:, :3]  # (N, 3)
        
        if self.use_color:
            colors = mesh[:, 3:6]
            data = np.concatenate([data, colors], 1)

        if self.use_normal:
            normals = mesh[:, 6:9]
            data = np.concatenate([data, normals], 1)
        
        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(self.cfg.SCANNETV2_PATH.multiview_features, "r", libver="latest")

            multiview = self.multiview_data[pid][scene_id]
            data = np.concatenate([data, multiview], 1)

        feats = data[:, 3:]
        points = mesh[:, :3]

        data = {"id": id, "scene_id": scene_id}
        # data = {"id": id}
        # data["id"] = np.array(id).astype(np.int32)
        # data["scene_id"] = np.array(int(scene_id.lstrip("scene").replace("_", ""))).astype(np.int32)

        if self.split != "test":
            instance_ids = scene["instance_ids"]
            sem_labels = scene["sem_labels"]  # {0,1,...,19}, -1 as ignored (unannotated) class
            
            # augment
            if self.split == "train" and self.cfg.general.task == "train":
                points_augment = self._augment(points)
            else:
                points_augment = points.copy()
            
            # scale
            points = points_augment * self.scale
            
            # elastic
            if self.split == "train" and self.cfg.general.task == "train":
                points = elastic(points, 6 * self.scale // 50,
                             40 * self.scale / 50)
                points = elastic(points, 20 * self.scale // 50,
                             160 * self.scale / 50)
            
            # offset
            points -= points.min(0)
            
            if self.split == "train" and self.cfg.general.task == "train":
                ### crop
                points, valid_idxs = crop(points, self.max_num_point, self.full_scale[1])
                # points, valid_idxs = random_sampling(points, self.max_num_point, return_choices=True)
                
                points = points[valid_idxs]
                points_augment = points_augment[valid_idxs]
                feats = feats[valid_idxs]
                sem_labels = sem_labels[valid_idxs]
                # instance_ids = instance_ids[valid_idxs]
                instance_ids = self._croppedInstanceIds(instance_ids, valid_idxs)
            
            if self.requires_bbox:
                num_instance, instance_info, instance_num_point, instance_bboxes, instance_bboxes_semcls, instance_bbox_ids, angle_classes, angle_residuals, size_classes, size_residuals, bbox_label = self._getInstanceInfo(points_augment, instance_ids, sem_labels)
            else:
                num_instance, instance_info, instance_num_point = self._getInstanceInfo(points_augment, instance_ids.astype(np.int32))

            data["locs"] = points_augment.astype(np.float32)  # (N, 3)
            data["locs_scaled"] = points.astype(np.float32)  # (N, 3)
            data["feats"] = feats.astype(np.float32)  # (N, 3)
            data["sem_labels"] = sem_labels.astype(np.int32)  # (N,)
            data["instance_ids"] = instance_ids.astype(np.int32)  # (N,) 0~total_nInst, -1
            data["num_instance"] = np.array(num_instance).astype(np.int32)  # int
            data["instance_info"] = instance_info.astype(np.float32)  # (N, 12)
            data["instance_num_point"] = np.array(instance_num_point).astype(np.int32)  # (num_instance,)
            if self.requires_bbox:
                data["center_label"] = instance_bboxes.astype(np.float32)[:,0:3] # (num_instance, 3) for GT box center XYZ
                data["sem_cls_label"] = instance_bboxes_semcls.astype(np.int64) # (num_instance,) semantic class index
                data["heading_class_label"] = angle_classes.astype(np.int64) # (num_instance,) with int values in 0,...,NUM_HEADING_BIN-1
                data["heading_residual_label"] = angle_residuals.astype(np.float32) # (num_instance,)
                data["size_class_label"] = size_classes.astype(np.int64) # (num_instance,) with int values in 0,...,NUM_SIZE_CLUSTER
                data["size_residual_label"] = size_residuals.astype(np.float32) # (num_instance, 3)
                data["gt_bbox_object_id"] = instance_bbox_ids.astype(np.int64)
                data["gt_bbox_label"] = bbox_label.astype(np.int64)
                data["gt_bbox"] = get_3d_box_batch(instance_bboxes[:, 0:3], instance_bboxes[:, 3:6], angle_classes).astype(np.float32) # (num_instance, 8, 3)
        else:
            # augment
            points_augment = points.copy()
            
            # scale
            points = points_augment * self.scale
            # points *= self.scale
            
            # offset
            points -= points.min(0)

            data["locs"] = points_augment.astype(np.float32)  # (N, 3)
            data["locs_scaled"] = points.astype(np.float32)  # (N, 3)
            data["feats"] = feats.astype(np.float32)  # (N, 3)

        return data
    

def scannet_loader(cfg):
    def scannet_collate_fn(batch):
        batch_size = batch.__len__()
        data = {}
        for key in batch[0].keys():
            if key in ['locs', 'locs_scaled', 'feats', 'sem_labels', 'instance_ids', 'num_instance', 'instance_info', 'instance_num_point']:
                continue
            if isinstance(batch[0][key], tuple):
                coords, feats = list(zip(*[sample[key] for sample in batch]))
                coords_b = batched_coordinates(coords)
                feats_b = torch.from_numpy(np.concatenate(feats, 0)).float()
                data[key] = (coords_b, feats_b)
            elif isinstance(batch[0][key], np.ndarray):
                data[key] = torch.stack(
                    [torch.from_numpy(sample[key]) for sample in batch],
                    axis=0)
            elif isinstance(batch[0][key], torch.Tensor):
                data[key] = torch.stack([sample[key] for sample in batch],
                                            axis=0)
            elif isinstance(batch[0][key], dict):
                data[key] = sparse_collate(
                    [sample[key] for sample in batch])
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
                instance_ids_i = b["instance_ids"]
                instance_ids_i[np.where(instance_ids_i != -1)] += total_num_inst
                total_num_inst += b["num_instance"].item()
                instance_ids.append(torch.from_numpy(instance_ids_i))
                
                sem_labels.append(torch.from_numpy(b["sem_labels"]))

                instance_info.append(torch.from_numpy(b["instance_info"]))
                instance_num_point.append(torch.from_numpy(b["instance_num_point"]))
                instance_offsets.append(instance_offsets[-1] + b["num_instance"].item())

        data["locs"] = torch.cat(locs, 0).to(torch.float32)  # float (N, 3)
        data["locs_scaled"] = torch.cat(locs_scaled, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        data["feats"] = torch.cat(feats, 0)  #.to(torch.float32)            # float (N, C)
        data["batch_offsets"] = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)
        
        if cfg.general.task != "test":
            data["sem_labels"] = torch.cat(sem_labels, 0).long()  # long (N,)
            data["instance_ids"] = torch.cat(instance_ids, 0).long()  # long, (N,)
            data["instance_info"] = torch.cat(instance_info, 0).to(torch.float32)  # float (total_nInst, 12)
            data["instance_num_point"] = torch.cat(instance_num_point, 0).int()  # (total_nInst)
            data["instance_offsets"] = torch.tensor(instance_offsets, dtype=torch.int)  # int (B+1)

        ### voxelize
        data["voxel_locs"], data["p2v_map"], data["v2p_map"] = pointgroup_ops.voxelization_idx(data["locs_scaled"], len(batch), 4) # mode=4

        return data

    if cfg.general.task == "train":
        splits = ["train", "val"]
    else:
        splits = [cfg.data.split]

    dataset = {split: ScanNet(cfg, split) for split in splits}

    dataloader = {
        split:
        DataLoader(dataset[split],
                    batch_size=cfg.data.batch_size,
                    shuffle=True if split == "train" else False,
                    pin_memory=True,
                    collate_fn=sparse_collate_fn) 
        for split in splits
    }

    return dataset, dataloader