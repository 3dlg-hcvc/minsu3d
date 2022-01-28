import os
import sys
import time
import h5py 
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from MinkowskiEngine.utils import sparse_collate, batched_coordinates

sys.path.append("../")  # HACK add the lib folder
from lib.pointgroup_ops.functions import pointgroup_ops
from lib.utils.pc import crop
from lib.utils.bbox import get_3d_box_batch
from lib.utils.transform import jitter, flip, rotz, elastic
from lib.dataset.scannet import ScanNet


class GTFeatureScanNet(ScanNet): 
    
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

        instance_ids = scene["instance_ids"]
        sem_labels = scene["sem_labels"]  # {0,1,...,19}, -1 as ignored (unannotated) class
        
        # augment
        if self.split == "train":
            points_augment, m = self._augment(points, return_mat=True)
        else:
            points_augment = points.copy()
            m = np.eye(3)
        
        # scale
        points = points_augment * self.scale
        
        # elastic
        if self.split == "train":
            points = elastic(points, 6 * self.scale // 50, 40 * self.scale / 50)
            points = elastic(points, 20 * self.scale // 50, 160 * self.scale / 50)
        
        # offset
        points -= points.min(0)
        
        if self.split == "train":
            ### crop
            points, valid_idxs = crop(points, self.max_num_point, self.full_scale[1])
            points = points[valid_idxs]
            points_augment = points_augment[valid_idxs]
            feats = feats[valid_idxs]
            sem_labels = sem_labels[valid_idxs]
            instance_ids = self._croppedInstanceIds(instance_ids, valid_idxs)
            
        gt_proposals_idx, gt_proposals_offset, object_ids, instance_bboxes = self._generate_gt_clusters(points, instance_ids)
        
        heading_angles = np.zeros((len(instance_bboxes),))
        bbox_corner = get_3d_box_batch(instance_bboxes[:, 0:3], instance_bboxes[:, 3:6], heading_angles).astype(np.float32)

        data["locs"] = points_augment.astype(np.float32)  # (N, 3)
        data["locs_scaled"] = points.astype(np.float32)  # (N, 3)
        data["feats"] = feats.astype(np.float32)  # (N, 3)
        data['gt_proposals_idx'] = gt_proposals_idx
        data['gt_proposals_offset'] = gt_proposals_offset
        data['bbox_corner'] = bbox_corner
        data['object_ids'] = np.array(object_ids).astype(np.int32)
        data['transformation'] = m

        return data
    

def scannet_gt_feats_loader(cfg):
    
    def scannet_collate_fn(batch):
        batch_size = batch.__len__()
        data = {}
        for key in batch[0].keys():
            if key in ['locs', 'locs_scaled', 'feats', 'gt_proposals_idx', 'gt_proposals_offset']:
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
        batch_offsets = [0]
        total_num_inst = 0
        total_points = 0
        
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
            
            total_num_inst += len(b["gt_proposals_offset"]) - 1
            total_points += len(b["locs"])

        data["locs"] = torch.cat(locs, 0).to(torch.float32)  # float (N, 3)
        data["locs_scaled"] = torch.cat(locs_scaled, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        data["feats"] = torch.cat(feats, 0)            # float (N, C)
        data["batch_offsets"] = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)
        data["gt_proposals_idx"] = torch.cat(gt_proposals_idx, 0).to(torch.int32)
        data["gt_proposals_offset"] = torch.cat(gt_proposals_offset, 0).to(torch.int32)

        ### voxelize
        data["voxel_locs"], data["p2v_map"], data["v2p_map"] = pointgroup_ops.voxelization_idx(data["locs_scaled"], len(batch), 4) # mode=4

        return data

    splits = ["train", "val"]

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