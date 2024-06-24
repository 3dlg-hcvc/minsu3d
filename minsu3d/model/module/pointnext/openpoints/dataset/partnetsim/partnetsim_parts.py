import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from ..data_util import crop_pc, voxelize
from ...transforms.point_transform_cpu import PointsToTensor
import glob
from tqdm import tqdm
import logging
import pickle


VALID_CLASS_IDS = [
    0, 1, 2, 3
]

PARTNETSIM_COLOR_MAP = {
    0: (255., 0, 0),
    1: (0, 255., 0),
    2: (0, 0, 255.),
    3: (255., 0, 255.),
}



@DATASETS.register_module()
class PartNetSim(Dataset):
    num_classes = 4
    classes = ["drawer", "door", "lid", "base"]
    gravity_dim = 2

    cls_parts = {"drawer": [0], "door": [1], "lid" : [2], "base": [3]}
    cls2parts = []
    cls2partembed = torch.zeros(16, 50)
    for i, cls in enumerate(classes):
        idx = cls_parts[cls]
        cls2parts.append(idx)
        cls2partembed[i].scatter_(0, torch.LongTensor(idx), 1)
    part2cls = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in cls_parts.keys():
        for label in cls_parts[cat]:
            part2cls[label] = cat
    
    #color_mean = [0.46259782, 0.46253258, 0.46253258]
    #color_std =  [0.693565  , 0.6852543 , 0.68061745]
    """ScanNet dataset, loading the subsampled entire room as input without block/sphere subsampling.
    number of points per room in average, median, and std: (145841.0, 158783.87179487178, 84200.84445829492)
    """  
    def __init__(self,
                 data_root='data/partnetsim',
                 split='train',
                 num_points=2048,
                 transform=None,
                 presample=False,
                 ):
        super().__init__()
        self.split = split
        self.transform = transform
        self.presample = presample
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
        filename = os.path.join(
            processed_root, f'partnetsim_{split}.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []
            for item in tqdm(self.data_list, desc=f'Loading PartNetSim {split} split'):
                data = torch.load(item)
                coord = data['xyz']
                feat = data['normal']
                label = data['sem_labels']
                seg_inst = data["instance_ids"]
                #coord, feat, label = crop_pc(
                #    coord, feat, label, self.split, self.voxel_size, self.voxel_max, variable=self.variable)
                cdata = np.hstack(
                    (coord, feat, np.expand_dims(label, -1), np.expand_dims(seg_inst, -1)).astype(np.float32))
                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")
            # median, average, std of number of points after voxel sampling for val set.
            # (100338.5, 109686.1282051282, 57024.51083415437)
            # before voxel sampling
            # (145841.0, 158783.87179487178, 84200.84445829492)
    def __getitem__(self, idx):
        data_idx = idx % len(self.data_list)
        if self.presample:
            coord, feat, label, seg_inst = np.split(self.data[data_idx], [3, 6, 7], axis=1)
        else:
            data_path = self.data_list[data_idx]
            data = torch.load(data_path)
            coord = data['xyz']
            feat = data['normal']
            label = data['sem_labels']
            seg_inst = data["instance_ids"]

        #feat = (feat + 1) * 127.5
        label = label.astype(np.long).squeeze()
        seg_inst = seg_inst.astype(np.long).squeeze()
        data = {'pos': coord.astype(np.float32), 'x': feat.astype(np.float32), 'cls': label, 'y': label}
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
        return np.shape(self.data_list)[0] * 6