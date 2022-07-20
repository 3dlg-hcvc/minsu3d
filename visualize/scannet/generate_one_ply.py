import hydra
from pathlib import Path
import os, sys
import argparse
from omegaconf import OmegaConf
from plyfile import PlyData
import argparse
from tqdm import tqdm

import numpy as np
import torch
import matplotlib.pyplot as plt
import colorsys
import random

# both functions referenced from https://github.com/choumin/ncolors/blob/master/ncolors.py
def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num 
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors
def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors 
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors

sys.path.append(os.getcwd())
sys.path.append('../..')
from data.scannet.model_util_scannet import SCANNET_COLOR_MAP
from lib.util.pc import write_ply_rgb, write_ply_colorful, write_ply_rgb_face
from lib.util.bbox import write_cylinder_bbox
CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160]
}
SEMANTIC_IDX2NAME = {
    1: 'cabinet',
    2: 'bed',
    3: 'chair',
    4: 'sofa',
    5: 'table',
    6: 'door',
    7: 'window',
    8: 'bookshelf',
    9: 'picture',
    10: 'counter',
    11: 'desk',
    12: 'curtain',
    13: 'refridgerator',
    14: 'shower curtain',
    15: 'toilet',
    16: 'sink',
    17: 'bathtub',
    18: 'otherfurniture'
}

def generate_single_ply(args):
    args.output_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(args.output_dir, exist_ok=True)
    rgb_file = os.path.join(args.rgb_file_dir, f'{args.scene_id}.pth')
    pred_sem_file = os.path.join(args.predict_dir, f'{args.scene_id}.txt')
    rgb_inst_ply = os.path.join(args.output_dir, f'{args.scene_id}.ply')

    scannet_data = torch.load(rgb_file)
    points = scannet_data['xyz'].astype(np.float32)
    colors = scannet_data['rgb'].astype(np.uint8)
    with open(pred_sem_file) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    instanceFileNames = []
    labelIndexes = []
    confidenceScores = []
    for i in lines:
        splitedLine = i.split()
        instanceFileNames.append(os.path.join(args.predict_dir, splitedLine[0]))
        labelIndexes.append(splitedLine[1])
        confidenceScores.append(splitedLine[2])
    if args.mode == "semantic":
        for index, instanceFileName in enumerate(instanceFileNames):
            with open(instanceFileName) as file:
                predicted_mask_list = file.readlines()
                predicted_mask_list = [line.rstrip() for line in predicted_mask_list]
            semanticIndex = labelIndexes[index]
            confidence = confidenceScores[index]
            for vertexIndex, color in enumerate(colors):
                if predicted_mask_list[vertexIndex] == "1":
                    colors[vertexIndex] = CLASS_COLOR[SEMANTIC_IDX2NAME[int(semanticIndex)]]
    elif args.mode == "instance":
        color_list = ncolors(len(labelIndexes))
        random.shuffle(color_list)
        for index, instanceFileName in enumerate(instanceFileNames):
            with open(instanceFileName) as file:
                predicted_mask_list = file.readlines()
                predicted_mask_list = [line.rstrip() for line in predicted_mask_list]
            for vertexIndex, color in enumerate(colors):
                if predicted_mask_list[vertexIndex] == "1":
                    colors[vertexIndex] = color_list[index]
    write_ply_rgb(points, colors, rgb_inst_ply)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predict_dir', type=str, default='/localhome/zmgong/research/pointgroup-minkowski/output/ScanNet/SoftGroup/test/predictions/instance', help='Spiciy the directory of the predictions')
    parser.add_argument('-r', '--rgb_file_dir', type=str, default='/localhome/zmgong/research/pointgroup-minkowski/data/scannet/splited_data/val')
    
    parser.add_argument('-sid', '--scene_id', type=str, default='scene0011_00', help='scene ID(example: scene0011_00)')
    # parser.add_argument('-sid', '--scene_id', type=str, required=True, help='scene ID(example: scene0011_00)')
    
    parser.add_argument('-m', '--mode', type=str, default='instance', choices=['semantic', 'instance'],help='specify instance or semantic mode: semantic | instance | detection')
    parser.add_argument('-t', '--type', type=str, default='pointcloud', help='specify type of ply: pointcloud | mesh')
    parser.add_argument('-o', '--output_dir', type=str, default='output_ply', help='Spiciy the directory of the output ply')
    args = parser.parse_args()
    generate_pred_inst_ply(args)
