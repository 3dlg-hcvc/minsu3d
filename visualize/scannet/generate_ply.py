import os, sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import random
import colorsys

sys.path.append(os.getcwd())
sys.path.append('../..')
from data.scannet.model_util_scannet import SCANNET_COLOR_MAP
from minsu3d.util.pc import write_ply_rgb, write_ply_colorful, write_ply_rgb_face
from minsu3d.util.bbox import write_cylinder_bbox
import torch


# The following two functions referenced from https://github.com/choumin/ncolors/blob/master/ncolors.py
def get_hls_colors(num):
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

def get_rgb_colors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors

def generate_single_ply(args):

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
                    colors[vertexIndex] = SCANNET_COLOR_MAP[int(semanticIndex)]
    elif args.mode == "instance":
        color_list = get_rgb_colors(len(labelIndexes))
        random.shuffle(color_list)
        for index, instanceFileName in enumerate(instanceFileNames):
            with open(instanceFileName) as file:
                predicted_mask_list = file.readlines()
                predicted_mask_list = [line.rstrip() for line in predicted_mask_list]
            for vertexIndex, color in enumerate(colors):
                if predicted_mask_list[vertexIndex] == "1":
                    colors[vertexIndex] = color_list[index]
    write_ply_rgb(points, colors, rgb_inst_ply)

def generate_pred_inst_ply(args):
    metadata_path = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data/scannet/meta_data')
    scene_ids_file = os.path.join(metadata_path, f'scannetv2_{args.split}.txt')

    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    for scene_id in tqdm(scene_ids):
        args.scene_id = scene_id
        generate_single_ply(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predict_dir', type=str, default='../../output/ScanNet/SoftGroup/test/predictions/instance', help='Spiciy the directory of the predictions. Eg:"../../output/ScanNet/SoftGroup/test/predictions/instanc"')
    parser.add_argument('-s', '--split', type=str, default='val', choices=['test', 'val'],help='specify the split of data: val | test')
    parser.add_argument('-m', '--mode', type=str, default='semantic', choices=['semantic', 'instance'],help='specify instance or semantic mode: semantic | instance | detection')
    parser.add_argument('-t', '--type', type=str, default='pointcloud', help='specify type of ply: pointcloud | mesh')
    parser.add_argument('-o', '--output_dir', type=str, default='output_ply', help='Spiciy the directory of the output ply')
    args = parser.parse_args()
    args.rgb_file_dir = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data/scannet', args.split)
    args.output_dir = os.path.join(args.output_dir, args.mode)
    generate_pred_inst_ply(args)
