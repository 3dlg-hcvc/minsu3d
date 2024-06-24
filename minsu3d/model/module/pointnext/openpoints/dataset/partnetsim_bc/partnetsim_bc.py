"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""
import os
import glob
import h5py
import numpy as np
import pickle
import logging
import ssl
import urllib
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive, check_integrity
from ..build import DATASETS
from ...transforms.point_transform_cpu import PointsToTensor


VALID_CLASS_IDS = [
    0, 1, 2, 3
]

PARTNETSIM_COLOR_MAP = {
    0: (0.0, 107.0, 164.0),
    1: (255.0, 128.0, 14.0),
    2: (200.0, 82.0, 0.0),
    3: (171.0, 171.0, 171.0),
}


@DATASETS.register_module()
class PartNetSim_BC(Dataset):
    num_classes = 4
    classes = ["drawer", "door", "lid", "base"]
    gravity_dim = 2

    cmap = [(0.0, 107.0, 164.0), (255.0, 128.0, 14.0), (200.0, 82.0, 0.0), (171.0, 171.0, 171.0)]
    
    #color_mean = [0.46259782, 0.46253258, 0.46253258]
    #color_std =  [0.693565  , 0.6852543 , 0.68061745]
    """ScanNet dataset, loading the subsampled entire room as input without block/sphere subsampling.
    number of points per room in average, median, and std: (145841.0, 158783.87179487178, 84200.84445829492)
    """  
    def __init__(self,
                 data_root='data/partnetsim',
                 split='train',
                 transform=None,
                 ):
        super().__init__()
        self.split = split
        self.transform = transform
        self.pipe_transform = PointsToTensor() 

        if split == "train" or split == 'val':
            self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
        elif split == 'trainval':
            self.data_list = glob.glob(os.path.join(
                data_root, "train", "*.pth")) + glob.glob(os.path.join(data_root, "val", "*.pth"))
        elif split == 'test':
            self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
        else:
            raise ValueError("no such split: {}".format(split))

        logging.info("Totally {} samples in {} set.".format(
            len(self.data_list), split))

        processed_root = os.path.join(data_root, 'processed')
        
    def __getitem__(self, idx):
        data_idx = idx % len(self.data_list)
        
        data_path = self.data_list[data_idx]
        data = torch.load(data_path)
        coord = data['xyz']
        feat = data['normal']
        label = data['sem_labels']
        label2 = data['boundary_labels']

        #cls = data['sem_labels']

        #feat = (feat + 1) * 127.5
        label = label.astype(np.long).squeeze()
        label2 = label2.astype(np.long).squeeze()
        data = {'pos': coord.astype(np.float32), 'x': feat.astype(np.float32), 'y1': label, "y2": label2}
        """debug 
        from openpoints.dataset import vis_multi_points
        import copy
        old_data = copy.deepcopy(data)
        if self.transform is not None:
            data = self.transform(data)
        data['pos'], data['x'], data['y'] = crop_pc(
            data['pos'], data['x'], data['y'], self.split, self.voxel_size, self.voxel_max,
            downsample=not self.presample, variable=self.variable)
            
        vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3]], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3]])
        """
        if self.transform is not None:
            data = self.transform(data)
        
        #if not self.presample: 
            #data['pos'], data['x'], data['y'] = crop_pc(
            #   data['pos'], data['x'], data['y'], self.split, self.voxel_size, self.voxel_max,
            #    downsample=not self.presample, variable=self.variable)
        
        data = self.pipe_transform(data)
         
        if 'heights' not in data.keys():
            data['heights'] =  data['pos'][:, self.gravity_dim:self.gravity_dim+1] - data['pos'][:, self.gravity_dim:self.gravity_dim+1].min()
        return data

    def __len__(self):
        return len(self.data_list)
    
    @property
    def num_classes(self):
        return np.max(self.label) + 1

    