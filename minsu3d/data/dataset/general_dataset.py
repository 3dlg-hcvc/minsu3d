import os
from tqdm import tqdm
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from minsu3d.util.pc import crop
from minsu3d.util.transform import jitter, flip, rotz, elastic


class GeneralDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.dataset_root_path = cfg.data.dataset_path
        self.file_suffix = cfg.data.file_suffix
        self.full_scale = cfg.data.full_scale
        self.scale = cfg.data.scale
        self.max_num_point = cfg.data.max_num_point
        self.data_map = {
            "train": cfg.data.metadata.train_list,
            "val": cfg.data.metadata.val_list,
            "test": cfg.data.metadata.test_list
        }
        self._load_from_disk()
        if cfg.model.model.use_multiview:
            self.multiview_hdf5_file = h5py.File(self.cfg.data.metadata.multiview_file, "r", libver="latest")

    def _load_from_disk(self):
        with open(self.data_map[self.split]) as f:
            self.scene_names = [line.strip() for line in f]
        self.scenes = []
        for scene_name in tqdm(self.scene_names, desc=f"Loading {self.split} data from disk"):
            scene_path = os.path.join(self.dataset_root_path, self.split, scene_name + self.file_suffix)
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
        while j < instance_ids.max():
            if np.count_nonzero(instance_ids == j) == 0:
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
        # (n, 3), float, (meanx, meany, meanz)
        instance_info = np.empty(shape=(xyz.shape[0], 3), dtype=np.float32)
        instance_cls = np.full(shape=unique_instance_ids.shape[0], fill_value=self.cfg.data.ignore_label, dtype=np.int8)
        for index, i in enumerate(unique_instance_ids):
            inst_i_idx = np.where(instance_ids == i)[0]

            # instance_info
            xyz_i = xyz[inst_i_idx]

            mean_xyz_i = xyz_i.mean(0)

            # offset
            instance_info[inst_i_idx] = mean_xyz_i

            # instance_num_point
            instance_num_point.append(inst_i_idx.size)

            # semantic label
            cls_idx = inst_i_idx[0]
            instance_cls[index] = sem_labels[cls_idx] - len(self.cfg.data.ignore_classes) if sem_labels[cls_idx] != self.cfg.data.ignore_label else sem_labels[cls_idx]
            # bounding boxes

        return num_instance, instance_info, instance_num_point, instance_cls

    def __getitem__(self, idx):
        scene_id = self.scene_names[idx]
        scene = self.scenes[idx]

        points = scene["xyz"]  # (N, 3)
        colors = scene["rgb"]  # (N, 3)
        normals = scene["normal"]
        if self.cfg.model.model.use_multiview:
            multiviews = self.multiview_hdf5_file[scene_id]
        instance_ids = scene["instance_ids"]
        sem_labels = scene["sem_labels"]
        data = {"scan_id": scene_id}

        # augment
        if self.split == "train":
            aug_matrix = self._get_augmentation_matrix()
            points = np.matmul(points, aug_matrix)
            normals = np.matmul(normals, np.transpose(np.linalg.inv(aug_matrix)))
            if self.cfg.data.augmentation.jitter_rgb:
                # jitter rgb
                colors += np.random.randn(3) * 0.1

        # scale
        scaled_points = points * self.scale

        # elastic
        if self.split == "train" and self.cfg.data.augmentation.elastic:
            scaled_points = elastic(scaled_points, 6 * self.scale // 50, 40 * self.scale / 50)
            scaled_points = elastic(scaled_points, 20 * self.scale // 50, 160 * self.scale / 50)

        # offset
        scaled_points -= scaled_points.min(axis=0)

        # crop
        if self.split == "train":
            # HACK, in case there are few points left
            max_tries = 10
            valid_idxs_count = 0
            valid_idxs = np.ones(shape=scaled_points.shape[0], dtype=np.bool)
            if valid_idxs.shape[0] > self.max_num_point:
                while max_tries > 0:
                    points_tmp, valid_idxs = crop(scaled_points, self.max_num_point, self.full_scale[1])
                    valid_idxs_count = np.count_nonzero(valid_idxs)
                    if valid_idxs_count >= 5000:
                        scaled_points = points_tmp
                        break
                    max_tries -= 1
                if valid_idxs_count < 5000:
                    raise Exception("Over-cropped!")

            scaled_points = scaled_points[valid_idxs]
            points = points[valid_idxs]
            normals = normals[valid_idxs]
            colors = colors[valid_idxs]
            if self.cfg.model.model.use_multiview:
                multiviews = np.asarray(multiviews)[valid_idxs]
            sem_labels = sem_labels[valid_idxs]
            instance_ids = self._get_cropped_inst_ids(instance_ids, valid_idxs)

        num_instance, instance_info, instance_num_point, instance_semantic_cls = self._get_inst_info(
            points, instance_ids, sem_labels)

        feats = np.zeros(shape=(len(scaled_points), 0), dtype=np.float32)
        if self.cfg.model.model.use_color:
            feats = np.concatenate((feats, colors), axis=1)
        if self.cfg.model.model.use_normal:
            feats = np.concatenate((feats, normals), axis=1)
        if self.cfg.model.model.use_multiview:
            feats = np.concatenate((feats, multiviews), axis=1)

        data["locs"] = points  # (N, 3)
        data["locs_scaled"] = scaled_points  # (N, 3)
        data["feats"] = feats  # (N, 3)
        data["sem_labels"] = sem_labels  # (N,)
        data["instance_ids"] = instance_ids  # (N,) 0~total_nInst, -1
        data["num_instance"] = np.array(num_instance, dtype=np.int32)  # int
        data["instance_info"] = instance_info  # (N, 12)
        data["instance_num_point"] = np.array(instance_num_point, dtype=np.int32)  # (num_instance,)
        data["instance_semantic_cls"] = instance_semantic_cls
        return data
