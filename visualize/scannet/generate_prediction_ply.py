import argparse
import math
import os
import random
import sys

sys.path.append('../..')
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm
from minsu3d.util.pc import write_ply_rgb_face
from minsu3d.util.bbox import write_cylinder_bbox


SCANNET_COLOR_MAP = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    13: (46.0, 85.0, 103.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}


def get_bbox(predicted_mask, points):
    x_min = None
    y_min = None
    z_min = None
    x_max = None
    y_max = None
    z_max = None
    for vertexIndex, xyz in enumerate(points):
        if predicted_mask[vertexIndex] == True:
            if x_min is None or xyz[0] < x_min:
                x_min = xyz[0]
            if y_min is None or xyz[1] < y_min:
                y_min = xyz[1]
            if z_min is None or xyz[2] < z_min:
                z_min = xyz[2]
            if x_max is None or xyz[0] > x_max:
                x_max = xyz[0]
            if y_max is None or xyz[1] > y_max:
                y_max = xyz[1]
            if z_max is None or xyz[2] > z_max:
                z_max = xyz[2]
    return x_min, x_max, y_min, y_max, z_min, z_max


def get_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return [r, g, b]


def get_random_rgb_colors(num):
    rgb_colors = [get_random_color() for _ in range(num)]
    return rgb_colors


def generate_colored_ply(args, predicted_mask_list, labelIndexes, points, colors, indices,
                         rgb_inst_ply):
    if args.mode == "semantic":
        for index, predicted_mask in enumerate(predicted_mask_list):
            semanticIndex = labelIndexes[index]
            # confidence = confidenceScores[index]
            for vertexIndex, color in enumerate(colors):
                if predicted_mask[vertexIndex] == True:
                    colors[vertexIndex] = SCANNET_COLOR_MAP[int(semanticIndex)]
    elif args.mode == "instance":
        color_list = get_random_rgb_colors(len(labelIndexes))
        random.shuffle(color_list)
        for index, predicted_mask in enumerate(predicted_mask_list):
            for vertexIndex, color in enumerate(colors):
                if predicted_mask[vertexIndex] == True:
                    colors[vertexIndex] = color_list[index]
    write_ply_rgb_face(points, colors, indices, rgb_inst_ply)
    return 0


def generate_bbox_ply(args, predicted_mask_list, labelIndexes, points, colors, indices, rgb_inst_ply):
    b_verts = []
    b_colors = []
    b_indices = []
    for index, predicted_mask in enumerate(predicted_mask_list):
        x_min, x_max, y_min, y_max, z_min, z_max = get_bbox(predicted_mask, points)
        currbbox = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0, x_max - x_min, y_max - y_min,
                    z_max - z_min]

        if args.mode == 'semantic':
            semanticIndex = labelIndexes[index]
            chooseColor = SCANNET_COLOR_MAP[int(semanticIndex)]
        else:
            color_list = get_random_rgb_colors(len(labelIndexes))
            random.shuffle(color_list)
            chooseColor = color_list[index]
        curr_verts, curr_colors, curr_indices = write_cylinder_bbox(np.array(currbbox), 0, None, color=chooseColor)
        curr_indices = np.array(curr_indices)
        curr_indices = curr_indices + len(b_verts)
        curr_indices = curr_indices.tolist()
        b_verts.extend(curr_verts)
        b_colors.extend(curr_colors)
        b_indices.extend(curr_indices)

    points = points.tolist()
    colors = colors.tolist()
    indices = indices.tolist()
    b_indices = np.array(b_indices)
    b_indices = b_indices + len(points)
    b_indices = b_indices.tolist()
    points.extend(b_verts)
    colors.extend(b_colors)
    indices.extend(b_indices)

    points = np.array(points)
    colors = np.array(colors)
    indices = np.array(indices)
    write_ply_rgb_face(points, colors, indices, rgb_inst_ply)
    return 0


def generate_single_ply(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # define position of necessary files
    ply_file = os.path.join(args.scans, args.scene_id, f'{args.scene_id}_vh_clean_2.ply')
    # alignment_file = os.path.join(args.scans, args.scene_id, f'{args.scene_id}.txt')
    pred_sem_file = os.path.join(args.predict_dir, f'{args.scene_id}.txt')

    # define where to output the ply file
    rgb_inst_ply = os.path.join(args.output_dir, f'{args.scene_id}.ply')

    # get mesh
    scannet_data = o3d.io.read_triangle_mesh(ply_file)
    scannet_data.compute_vertex_normals()
    points = np.asarray(scannet_data.vertices)
    colors = np.asarray(scannet_data.vertex_colors)
    indices = np.asarray(scannet_data.triangles)
    colors = colors * 255.0

    with open(pred_sem_file) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    instanceFileNames = []
    labelIndexes = []
    confidenceScores = []
    predicted_mask_list = []
    for i in lines:
        splitedLine = i.split()
        instanceFileNames.append(os.path.join(args.predict_dir, splitedLine[0]))
        labelIndexes.append(splitedLine[1])
        confidenceScores.append(splitedLine[2])

    for instanceFileName in instanceFileNames:
        predicted_mask_list.append(np.loadtxt(instanceFileName, dtype=bool))

    if not args.bbox:
        generate_colored_ply(args, predicted_mask_list, labelIndexes, points, colors, indices,
                             rgb_inst_ply)
    else:
        generate_bbox_ply(args, predicted_mask_list, labelIndexes, points, colors, indices,
                          rgb_inst_ply)


def generate_pred_inst_ply(args):
    metadata_path = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data/scannet/metadata')
    scene_ids_file = os.path.join(metadata_path, f'scannetv2_{args.split}.txt')
    args.scans = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data/scannet/scans')

    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    for scene_id in tqdm(scene_ids):
        args.scene_id = scene_id
        generate_single_ply(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--predict_dir', type=str,
                        default='../../output/ScanNet/SoftGroup/test/predictions/instance',
                        help='the directory of the predictions. Eg:"../../output/ScanNet/SoftGroup/test/predictions/instanc"')
    parser.add_argument('-s', '--split', type=str, default='val', choices=['test', 'val'],
                        help='specify the split of data: val | test')
    parser.add_argument('-b', '--bbox', action='store_true',
                        help='specify to generate ply with bounding box or colored object')
    parser.set_defaults(bbox=False)
    parser.add_argument('-m', '--mode', type=str, default='semantic', choices=['semantic', 'instance'],
                        help='specify instance or semantic mode: semantic | instance')
    parser.add_argument('-o', '--output_dir', type=str, default='output_ply',
                        help='the directory of the output ply')
    args = parser.parse_args()
    args.rgb_file_dir = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data/scannet', args.split)
    if args.bbox == True:
        args.output_dir = os.path.join(args.output_dir, "bbox")
    else:
        args.output_dir = os.path.join(args.output_dir, "color")
    args.output_dir = os.path.join(args.output_dir, args.mode)

    generate_pred_inst_ply(args)
