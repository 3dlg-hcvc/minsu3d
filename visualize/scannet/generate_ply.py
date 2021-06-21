import os, sys
from omegaconf import OmegaConf
from plyfile import PlyData
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
# sys.path.append("../..")
from data.scannet.model_util_scannet import SCANNET_COLOR_MAP
from lib.utils.pc import write_ply_colorful, write_ply_rgb_face
from lib.utils.bbox import write_cylinder_bbox


cfg = OmegaConf.load('conf/path.yaml')


def generate_gt_sem_ply():
    from data.scannet.model_util_scannet import NYU20_CLASS_IDX
    split = 'val'
    data_dir = cfg.SCANNETV2_PATH.splited_scans
    output_dir = f'/local-scratch/qiruiw/dataset/scannet/splited_nyu20_labels/{split}'
    os.makedirs(output_dir, exist_ok=True)
    scene_ids_file = os.path.join(cfg.SCANNETV2_PATH.meta_data, f'scannetv2_{split}.txt')
    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    
    valid_class_idx = np.array(NYU20_CLASS_IDX[1:])
    
    for scene_id in scene_ids:
        print(scene_id)
        nyu40_sem_ply = os.path.join(data_dir, split, scene_id, f'{scene_id}_vh_clean_2.labels.ply')
        nyu20_sem_ply = os.path.join(output_dir, f'{scene_id}.ply')
        
        with open(nyu40_sem_ply, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            points = np.zeros(shape=[num_verts, 3], dtype=np.float)
            points[:,0] = plydata['vertex'].data['x']
            points[:,1] = plydata['vertex'].data['y']
            points[:,2] = plydata['vertex'].data['z']
            sem_labels = np.array(plydata['vertex']['label'])
            valid_label_idx = np.in1d(sem_labels, valid_class_idx)
            sem_labels[valid_label_idx] = 0
            
        write_ply_colorful(points, sem_labels, nyu20_sem_ply, colormap=SCANNET_COLOR_MAP)


def generate_pred_sem_ply():
    split = 'val'
    data_dir = cfg.SCANNETV2_PATH.splited_scans
    pred_dir = f'/local-scratch/qiruiw/research/dense-scanrefer/log/scannet/pointgroup/test/2021-02-10_01-53-53/splited_pred/{split}/semantic'
    scene_ids_file = os.path.join(cfg.SCANNETV2_PATH.meta_data, f'scannetv2_{split}.txt')
    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    
    for scene_id in scene_ids:
        print(scene_id)
        gt_sem_ply = os.path.join(data_dir, split, scene_id, f'{scene_id}_vh_clean_2.labels.ply')
        pred_sem_ply = os.path.join(pred_dir, f'{scene_id}.ply')
        pred_sem_labels = np.loadtxt(os.path.join(pred_dir, f'{scene_id}.txt'))
        
        with open(gt_sem_ply, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            assert num_verts == len(pred_sem_labels)
            points = np.zeros(shape=[num_verts, 3], dtype=np.float)
            points[:,0] = plydata['vertex'].data['x']
            points[:,1] = plydata['vertex'].data['y']
            points[:,2] = plydata['vertex'].data['z']
            
        write_ply_colorful(points, pred_sem_labels, pred_sem_ply, colormap=SCANNET_COLOR_MAP)
            

if __name__ == "__main__":
    generate_gt_sem_ply()
    # generate_pred_sem_ply()
    # generate_gt_alignments_ply()