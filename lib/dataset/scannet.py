import os
import sys
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append("../")  # HACK add the lib folder

from lib.util.pc import crop
from lib.util.transform import jitter, flip, rotz, elastic


class ScanNet(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.root = cfg.SCANNETV2_PATH.splited_data
        self.file_suffix = cfg.data.file_suffix

        self.full_scale = cfg.data.full_scale
        self.scale = cfg.data.scale
        self.max_num_point = cfg.data.max_num_point

        self.DATA_MAP = {
            "train": cfg.SCANNETV2_PATH.train_list,
            "val": cfg.SCANNETV2_PATH.val_list,
            "test": cfg.SCANNETV2_PATH.test_list
        }

        self._load_from_disk()

    def _load_from_disk(self):
        with open(self.DATA_MAP[self.split]) as f:
            self.scene_names = [line.strip() for line in f]
        self.scenes = []
        for scene_name in tqdm(self.scene_names, desc="Loading data from disk"):
            scene_path = os.path.join(self.root, self.split, scene_name + self.file_suffix)
            scene = torch.load(scene_path)
            scene["xyz"] -= scene["xyz"].mean(axis=0)
            scene["rgb"] = scene["rgb"].astype(np.float32) / 127.5 - 1
            self.scenes.append(scene)

    def __len__(self):
        return len(self.scenes)

    def _get_augmentation_matrix(self):
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
        return m.astype(np.float32)

    def _get_cropped_inst_ids(self, instance_ids, valid_idxs):
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

    def _get_inst_info(self, xyz, instance_ids, sem_labels):
        """
        :param xyz: (n, 3)
        :param instance_ids: (n), int, (0~nInst-1, -1)
        :return: num_instance, dict
        """
        instance_num_point = []  # (nInst), int
        unique_instance_ids = np.unique(instance_ids)
        unique_instance_ids = unique_instance_ids[unique_instance_ids != self.cfg.data.ignore_label]
        num_instance = unique_instance_ids.shape[0]
        # (n, 12), float, (meanx, meany, meanz, cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_info = np.zeros(shape=(xyz.shape[0], 3), dtype=np.float32)
        instance_cls = np.full(shape=unique_instance_ids.shape[0], fill_value=self.cfg.data.ignore_label, dtype=np.int8)
        instance_bboxes = np.full(shape=(unique_instance_ids.shape[0], 6), fill_value=self.cfg.data.ignore_label, dtype=np.float32)
        for index, i in enumerate(unique_instance_ids):

            inst_i_idx = np.where(instance_ids == i)
            # instance_info
            xyz_i = xyz[inst_i_idx]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            # c_xyz_i = (max_xyz_i + min_xyz_i) / 2

            # offset
            instance_info[inst_i_idx] = mean_xyz_i

            # instance_info_i[:, 3:6] = c_xyz_i
            # instance_info_i[:, 3:6] = min_xyz_i
            # instance_info_i[:, 6:9] = max_xyz_i

            # instance_num_point
            instance_num_point.append(inst_i_idx[0].size)

            # semantic label
            cls_idx = inst_i_idx[0][0]
            assert sem_labels[cls_idx] not in self.cfg.data.ignore_classes
            instance_cls[index] = sem_labels[cls_idx] - len(self.cfg.data.ignore_classes) if sem_labels[cls_idx] != self.cfg.data.ignore_label else sem_labels[cls_idx]
            # bounding boxes
            instance_bboxes[index] = np.concatenate((min_xyz_i, max_xyz_i))

        return num_instance, instance_info, instance_num_point, instance_cls, instance_bboxes

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

    def __getitem__(self, idx):
        scan_id = self.scene_names[idx]
        scene = self.scenes[idx]

        points = scene["xyz"]  # (N, 3)
        colors = scene["rgb"]  # (N, 3) rgb
        normals = scene["normal"]

        data = {"scan_id": scan_id}

        if self.split != "test":
            instance_ids = scene["instance_ids"]
            sem_labels = scene["sem_labels"]  # {0,1,...,19}, -1 as ignored (unannotated) class
            # augment
            if self.split == "train":
                aug_matrix = self._get_augmentation_matrix()
                points_augment = np.matmul(points, aug_matrix)
                normals = np.matmul(normals, np.transpose(np.linalg.inv(aug_matrix)))
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
                colors += np.random.randn(3) * 0.1

            # offset
            points -= points.min(axis=0)

            if self.split == "train":
                # crop
                # HACK, in case there are few points left
                max_tries = 10
                valid_idxs_count = 0
                while max_tries > 0:
                    points_tmp, valid_idxs = crop(points, self.max_num_point, self.full_scale[1])
                    valid_idxs_count = np.count_nonzero(valid_idxs)
                    if valid_idxs_count >= 5000:
                        points = points_tmp
                        break
                    max_tries -= 1
                if valid_idxs_count < 5000:
                    raise Exception("Over-cropped!")
                # points, valid_idxs = random_sampling(points, self.max_num_point, return_choices=True)

                points = points[valid_idxs]
                points_augment = points_augment[valid_idxs]
                normals = normals[valid_idxs]
                colors = colors[valid_idxs]
                sem_labels = sem_labels[valid_idxs]
                instance_ids = self._get_cropped_inst_ids(instance_ids, valid_idxs)

            num_instance, instance_info, instance_num_point, instance_semantic_cls, instance_bboxes = self._get_inst_info(
                points_augment, instance_ids.astype(np.int32), sem_labels)


            feats = np.zeros(shape=(len(points), 0), dtype=np.float32)
            if self.cfg.model.model.use_color:
                feats = np.concatenate((feats, colors), axis=1)
            if self.cfg.model.model.use_normal:
                feats = np.concatenate((feats, normals), axis=1)
            data["locs"] = points_augment  # (N, 3)
            data["locs_scaled"] = points  # (N, 3)
            data["feats"] = feats  # (N, 3)
            data["sem_labels"] = sem_labels  # (N,)
            data["instance_ids"] = instance_ids  # (N,) 0~total_nInst, -1
            data["num_instance"] = np.array(num_instance, dtype=np.int32)  # int
            data["instance_info"] = instance_info  # (N, 12)
            data["instance_num_point"] = np.array(instance_num_point, dtype=np.int32)  # (num_instance,)
            data["instance_semantic_cls"] = instance_semantic_cls
            data["instance_bboxes"] = instance_bboxes

        else:
            # scale
            points = points * self.scale

            # offset
            points -= points.min(0)

            data["locs"] = points  # (N, 3)
            data["locs_scaled"] = points  # (N, 3)
            feats = np.zeros(shape=(len(points), 0), dtype=np.float32)
            if self.cfg.model.model.use_color:
                feats = np.concatenate((feats, colors), axis=1)
            if self.cfg.model.model.use_normal:
                feats = np.concatenate((feats, normals), axis=1)
            data["feats"] = feats  # (N, 3)

        return data
