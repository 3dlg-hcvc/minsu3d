import os
import argparse
from omegaconf import OmegaConf
from plyfile import PlyData
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors
from plyfile import PlyData, PlyElement
import colorsys

MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
color_cache = {}
sem_color_cache = {}

colors  = ["#fabed4", "#FF6D00", "#00C853", "#0091EA",  "#00ced1", "#D50000", "#FFD600",
              "#673AB7", "#3F51B5", "#795548", "#AEEA00", "#009688", "#ba55d3", "#e9967a", "#607D8B", "#E91E63",
              "#7fff00", "#AA00FF"]
colors = [list(matplotlib.colors.to_rgb(color)) for color in colors]


def increase_s(r, g, b, adjust_value):
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, l, s+adjust_value)


def get_inst_color(sem_label, color_len):
    if sem_label not in sem_color_cache:
        sem_color_cache[sem_label] = colors[sem_label - 1].copy()
        color = sem_color_cache[sem_label]
    else:
        adjust_value = +0.5 / color_len if color_len > 1 else 0
        sem_color_cache[sem_label][0], sem_color_cache[sem_label][1], sem_color_cache[sem_label][2] = increase_s(sem_color_cache[sem_label][0], sem_color_cache[sem_label][1], sem_color_cache[sem_label][2], adjust_value)
        color = sem_color_cache[sem_label]
    
    for i in range(len(color)):
        color[i] = min(1, color[i])
        color[i] = max(0, color[i])
    #color_cache[(sem_label, instance_id)] = color
    return color

def write_ply_rgb(points, colors, filename, plydata, text=True, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as ply file """
    colors = colors.astype(int)
    points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    ele = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([ele, plydata.elements[1]], text=text).write(filename)


def write_ply_colorful(points, labels, filename, plydata):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as ply file """
    labels = labels.astype(int)
    N = points.shape[0]
    
    vertex = []
    
    for i in range(N):
        # import pdb; pdb.set_trace()
        if labels[i] > 0:
            c = cm.get_cmap("tab20")(labels[i])[:3]
            c = [i *  255 for i in c]
        else:
            c =  (plydata.elements[0][i]['red'], plydata.elements[0][i]['green'], plydata.elements[0][i]['blue'])
        vertex.append( (points[i,0],points[i,1],points[i,2],c[0], c[1],c[2]) )
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])

    
    ele1 = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([ele1, plydata.elements[1]], text=True).write(filename)


def generate_pred_inst_ply():
    split = "val"
    # scan_dir = cfg.SCANNETV2_PATH.splited_scans
    data_dir = "/localhome/yza440/Research/pointgroup-minkowski/data/multiscan_inst/val"
    pred_dir = f'/localhome/yza440/Research/pointgroup-minkowski_old/log/multiscan/pointgroup/test/110000_inst/splited_pred' # TODO
    output_dir = f'/localhome/yza440/vis_inst' # TODO
    os.makedirs(output_dir, exist_ok=True)
    scene_ids_file = "/localhome/yza440/Research/multiscan/data_preparation/split_info/inst_seg_task_val.txt"
    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    global color_cache
    global sem_color_cache
    for scene_id in scene_ids:
        print(scene_id)
        color_cache = {}
        sem_color_cache = {}
        # rgb_ply = os.path.join(scan_dir, split, scene_id, f'{scene_id}_vh_clean_2.ply')
        rgb_file = os.path.join(data_dir, f'{scene_id}.pth')
        # pred_sem_file = os.path.join(pred_dir, split, f'semantic/{scene_id}.txt')
        pred_inst_file = os.path.join(pred_dir, split, f'instance/{scene_id}.txt')
        
        # original_mesh_file = os.path.join(cfg.MULTISCAN_PATH.raw_scans, scene_id, scene_id + "_clean.ply")
        # tmp_scene_id = scene_id.rsplit('_', 1)[0]
        original_mesh_file = os.path.join('/localhome/yza440/Research/multiscan_dataset/', scene_id, scene_id + "_clean.ply")

        plydata = PlyData.read(original_mesh_file)

        # rgb_data = PlyData.read(open(rgb_ply, 'rb'))
        scannet_data = torch.load(rgb_file)
        points = scannet_data['coords'].astype("f4")
        colors = np.ones_like(points) * 153
        num_verts = len(points)
        instance_ids = np.full(shape=num_verts, fill_value=-1, dtype=np.int32)
        with open(pred_inst_file, "r") as f:
            for i, line in enumerate(f):
                line_ele = line.strip().split()
                obj_pred_file = os.path.join(pred_dir, split, 'instance', line_ele[0])
                mask = np.loadtxt(obj_pred_file, dtype=bool)
                instance_ids[mask] = i
        rgb_inst_ply = os.path.join(output_dir, f'{scene_id}.ply')
        # num_verts = rgb_data['vertex'].count
        # sem_labels = np.loadtxt(pred_sem_file, dtype=np.int32)
        #instance_ids = np.loadtxt(pred_inst_file, dtype=np.int)
        # assert num_verts == len(sem_labels) and num_verts == len(instance_ids)
        
        # points = np.zeros(shape=[num_verts, 3], dtype=np.float)
        # colors = np.zeros(shape=[num_verts, 3], dtype=np.uint)
        # points[:,0] = rgb_data['vertex'].data['x']
        # points[:,1] = rgb_data['vertex'].data['y']
        # points[:,2] = rgb_data['vertex'].data['z']
        # colors[:,0] = rgb_data['vertex'].data['red']
        # colors[:,1] = rgb_data['vertex'].data['green']
        # colors[:,2] = rgb_data['vertex'].data['blue']
     
        unique_inst_ids = np.unique(instance_ids)
     
        colormap = [plt.cm.rainbow(i/(len(unique_inst_ids)+1)) for i in range(len(unique_inst_ids))]
        random.shuffle(colormap)
        for i, inst_id in enumerate(unique_inst_ids):
            if inst_id == -1: continue
            inst_vert_idx = instance_ids == inst_id
            # assert len(np.unique(sem_labels[inst_vert_idx])) == 1
            
            colors[inst_vert_idx, :] = np.array(colormap[i][:3]) * 255
           
        write_ply_rgb(points, colors, rgb_inst_ply, plydata)


def generate_gt_inst_ply():
    inst_data_dir = "/localhome/yza440/Research/pointgroup-minkowski/data/multiscan_inst/val"
    output_dir = f'/local-scratch/localhome/yza440/Research/pointgroup-minkowski/vis_gt' # TODO
    os.makedirs(output_dir, exist_ok=True)
    scene_ids_file = '/localhome/yza440/Research/pointgroup-minkowski/data/multiscan_inst/meta_data/inst_seg_task_val.txt'
    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    DONOTCARE_CLASS_IDS = np.array([-1, 0, 1, 2]) # exclude wall, floor and ceiling
    global color_cache
    global sem_color_cache
    for scene_id in scene_ids:
        # print(scene_id)
        color_cache = {}
        sem_color_cache = {}
        # rgb_ply = os.path.join(scan_dir, split, scene_id, f'{scene_id}_vh_clean_2.ply')
        gt_inst_file = os.path.join(inst_data_dir, f'{scene_id}.pth')
        rgb_inst_ply = os.path.join(output_dir, f'{scene_id}.ply')
        
        tmp_scene_id = scene_id.rsplit('_', 1)[0]
        original_mesh_file = os.path.join('/localhome/yza440/Research/multiscan_dataset/', scene_id, scene_id + "_clean.ply")

        plydata = PlyData.read(original_mesh_file)

        # rgb_data = PlyData.read(open(rgb_ply, 'rb'))
        scannet_data = torch.load(gt_inst_file)
        points = scannet_data['coords'].astype("f4")
        colors = np.ones_like(points) * 153
        num_verts = len(points)
        instance_ids = scannet_data['instance_ids']

        sem_labels = scannet_data['sem_labels']
   
        assert num_verts == len(sem_labels) and num_verts == len(instance_ids)
        
        invalid_vert_idx = np.in1d(sem_labels, DONOTCARE_CLASS_IDS)
        instance_ids[invalid_vert_idx] = -1 # -1: points are not assigned to any objects
        
        # points = np.zeros(shape=[num_verts, 3], dtype=np.float)
        # colors = np.zeros(shape=[num_verts, 3], dtype=np.uint)
        # points[:,0] = rgb_data['vertex'].data['x']
        # points[:,1] = rgb_data['vertex'].data['y']
        # points[:,2] = rgb_data['vertex'].data['z']
        # colors[:,0] = rgb_data['vertex'].data['red']
        # colors[:,1] = rgb_data['vertex'].data['green']
        # colors[:,2] = rgb_data['vertex'].data['blue']
        
        unique_inst_ids = np.unique(instance_ids)
    
        colormap = [plt.cm.rainbow(i/(len(unique_inst_ids)+1)) for i in range(len(unique_inst_ids))]
        random.shuffle(colormap)

        for i, inst_id in enumerate(unique_inst_ids):
            if inst_id == -1: continue
            inst_vert_idx = instance_ids == inst_id
            assert len(np.unique(sem_labels[inst_vert_idx])) == 1
            colors[inst_vert_idx, :] = np.array(colormap[i][:3]) * 255
        
        write_ply_rgb(points, colors, rgb_inst_ply, plydata)
        print(scene_id + "," + str(num_verts))



def generate_pred_sem_ply(args, cfg):
    split = "val"
    pred_dir = f'/localhome/yza440/pointgroup-minkowski/log/multiscan/pointgroup/test/2022-03-27_20-31-46/splited_pred/val/semantic' # TODO
    scene_ids_file = os.path.join(cfg.MULTISCAN_PATH.meta_data, f'456.txt')
    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    
    for scene_id in scene_ids:
        print(scene_id)
        gt_sem_ply = os.path.join('/project/3dlg-hcvc/multiscan/www/part_task_vis/single_objs_gt/mesh/ply', scene_id + ".ply")
        output_dir = f'/local-scratch/localhome/yza440/pointgroup-minkowski/vis'
        pred_sem_labels = np.loadtxt(os.path.join(pred_dir, f'{scene_id}.txt'))
        pred_sem_ply = os.path.join(output_dir, f'{scene_id}.ply')
        with open(gt_sem_ply, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            assert num_verts == len(pred_sem_labels)
            points = np.zeros(shape=[num_verts, 3], dtype=np.float)
            points[:,0] = plydata['vertex'].data['x']
            points[:,1] = plydata['vertex'].data['y']
            points[:,2] = plydata['vertex'].data['z']
            
        write_ply_colorful(points, pred_sem_labels, pred_sem_ply, plydata)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Visualize the pred ply")
    # parser.add_argument('-c', '--config', type=str, default='conf/pointgroup_multiscan.yaml', help='path to config file')
    # args = parser.parse_args()

    # base_cfg = OmegaConf.load('conf/path.yaml')
    # cfg = OmegaConf.load(args.config)
    # cfg = OmegaConf.merge(base_cfg, cfg)

    generate_pred_inst_ply()
    # generate_gt_inst_ply()
    # generate_pred_sem_ply(args, cfg)
