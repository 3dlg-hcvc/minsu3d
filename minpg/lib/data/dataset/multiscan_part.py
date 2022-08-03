from minpg.lib.data.dataset import GeneralDataset
from tqdm import tqdm
import torch
import numpy as np
import os


class MultiScanPart(GeneralDataset):
    def __int__(self, cfg, split):
        super().__init__(cfg, split)

    def _load_from_disk(self):
        with open(self.data_map[self.split]) as f:
            self.scene_names = [line.strip() for line in f]
        self.scenes = []
        for scene_name in tqdm(self.scene_names, desc=f"Loading {self.split} data from disk"):
            scene_path = os.path.join(self.dataset_root_path, self.split, scene_name + self.file_suffix)
            scene = torch.load(scene_path)
            scene["xyz"] = scene["coords"] - scene["coords"].mean(axis=0)
            scene["rgb"] = scene["colors"].astype(np.float32) / 127.5 - 1
            scene["normal"] = scene["normals"]
            del scene["coords"]
            del scene["colors"]
            del scene["normals"]
            self.scenes.append(scene)
