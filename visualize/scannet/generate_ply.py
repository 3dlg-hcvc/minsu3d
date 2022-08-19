import os, sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import random
import colorsys
import math
import torch
import open3d as o3d

sys.path.append(os.getcwd())
sys.path.append('../..')
from data.scannet.model_util_scannet import SCANNET_COLOR_MAP
from data.scannet.prepare_all_data import read_axis_align_matrix, read_mesh_file
from minsu3d.util.pc import write_ply_rgb, write_ply_colorful, write_ply_rgb_face
from minsu3d.util.bbox import write_cylinder_bbox
from minsu3d.util.bbox import write_bbox, write_cylinder_bbox, write_cylinder_bbox_batch

allColorList = []


def initialColor():
    for r in range(255):
        for g in range(255):
            for b in range(255):
                allColorList.append([r, g, b])


def get_random_rgb_colors(num):
    numberToSkip = math.floor(len(allColorList) / num)
    rgb_colors = allColorList[::numberToSkip]
    random.shuffle(rgb_colors)
    return rgb_colors


def non_maximum_suppression(instanceFileNames, labelIndexes, confidenceScores):
    list_of_all_predicted_mask_list = []
    for index, instanceFileName in enumerate(instanceFileNames):
        with open(instanceFileName) as file:
            predicted_mask_list = file.readlines()
            predicted_mask_list = [line.rstrip() for line in predicted_mask_list]
            list_of_all_predicted_mask_list.append(predicted_mask_list)
    pair_of_ij = []
    for i in range(len(list_of_all_predicted_mask_list)):
        for j in range(i + 1, len(list_of_all_predicted_mask_list)):
            mask1 = list_of_all_predicted_mask_list[i]
            mask2 = list_of_all_predicted_mask_list[j]
            count_of_both_one = 0.0
            count_of_single_one = 0.0
            for index in range(len(mask1)):
                if mask1[index] != str(0) and mask2[index] != str(0):
                    count_of_both_one = count_of_both_one + 1.0
                if mask1[index] != str(0) or mask2[index] != str(0):
                    count_of_single_one = count_of_single_one + 1.0
            if count_of_both_one / count_of_single_one > 0.65:
                pair_of_ij.append([i, j])
    res = [True for i in range(len(instanceFileNames))]
    for x in pair_of_ij:
        i = x[0]
        j = x[1]
        if confidenceScores[i] > confidenceScores[j]:
            res[j] = False
        else:
            res[i] = False
    newInstanceFileName = []
    newLabelIndexs = []
    newConfidenceScors = []
    for i in range(len(res)):
        result = res[i]
        if result:
            newInstanceFileName.append(instanceFileNames[i])
            newLabelIndexs.append(labelIndexes[i])
            newConfidenceScors.append(confidenceScores[i])

    return newInstanceFileName, newLabelIndexs, newConfidenceScors


def generate_colored_ply(args, pred_sem_file, points, colors, indices, rgb_inst_ply):
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
    if args.nms:
        instanceFileNames, labelIndexes, confidenceScores = non_maximum_suppression(instanceFileNames, labelIndexes,
                                                                                    confidenceScores)

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
        color_list = get_random_rgb_colors(len(labelIndexes))
        random.shuffle(color_list)
        for index, instanceFileName in enumerate(instanceFileNames):
            with open(instanceFileName) as file:
                predicted_mask_list = file.readlines()
                predicted_mask_list = [line.rstrip() for line in predicted_mask_list]
            for vertexIndex, color in enumerate(colors):
                if predicted_mask_list[vertexIndex] == "1":
                    colors[vertexIndex] = color_list[index]
    write_ply_rgb_face(points, colors, indices, rgb_inst_ply)
    return 0


def generate_bbox_ply(args, pred_sem_file, points, colors, indices, rgb_inst_ply):
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
    if args.nms:
        instanceFileNames, labelIndexes, confidenceScores = non_maximum_suppression(instanceFileNames, labelIndexes,
                                                                                    confidenceScores)
    b_verts = []
    b_colors = []
    b_indices = []
    for index, instanceFileName in enumerate(instanceFileNames):
        x_min = None
        y_min = None
        z_min = None
        x_max = None
        y_max = None
        z_max = None
        with open(instanceFileName) as file:
            predicted_mask_list = file.readlines()
            predicted_mask_list = [line.rstrip() for line in predicted_mask_list]
        for vertexIndex, xyz in enumerate(points):
            if predicted_mask_list[vertexIndex] == "1":
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

        currBbox = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0, x_max - x_min, y_max - y_min,
                    z_max - z_min]
        if args.mode == 'semantic':
            semanticIndex = labelIndexes[index]
            chooseColor = SCANNET_COLOR_MAP[int(semanticIndex)]
        else:
            color_list = get_randome_rgb_colors(len(labelIndexes))
            random.shuffle(color_list)
            chooseColor = color_list[index]
        curr_verts, curr_colors, curr_indices = write_cylinder_bbox(np.array(currBbox), 0, None, color=chooseColor)
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
    rgb_file = os.path.join(args.rgb_file_dir, f'{args.scene_id}.pth')
    ply_file = os.path.join(args.scans, args.scene_id, f'{args.scene_id}_vh_clean_2.ply')
    meta_file = os.path.join(args.scans, args.scene_id, f'{args.scene_id}.txt')
    pred_sem_file = os.path.join(args.predict_dir, f'{args.scene_id}.txt')

    # define where to output the ply file
    rgb_inst_ply = os.path.join(args.output_dir, f'{args.scene_id}.ply')

    # get mesh
    axis_align_matrix = read_axis_align_matrix(meta_file)
    scannet_data = o3d.io.read_triangle_mesh(ply_file)
    if axis_align_matrix is not None:
        # align the mesh
        scannet_data.transform(axis_align_matrix)
    scannet_data.compute_vertex_normals()
    points = np.asarray(scannet_data.vertices)
    colors = np.asarray(scannet_data.vertex_colors)
    indices = np.asarray(scannet_data.triangles)
    colors = colors * 255.0

    if args.bbox == False:
        generate_colored_ply(args, pred_sem_file, points, colors, indices, rgb_inst_ply)
    else:

        generate_bbox_ply(args, pred_sem_file, points, colors, indices, rgb_inst_ply)


def generate_pred_inst_ply(args):
    metadata_path = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data/scannet/meta_data')
    scene_ids_file = os.path.join(metadata_path, f'scannetv2_{args.split}.txt')
    args.scans = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data/scannet/scans')

    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    for scene_id in scene_ids:
        args.scene_id = scene_id
        generate_single_ply(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--predict_dir', type=str,
                        default='../../output/ScanNet/SoftGroup/test/predictions/instance',
                        help='Spiciy the directory of the predictions. Eg:"../../output/ScanNet/SoftGroup/test/predictions/instanc"')
    parser.add_argument('-s', '--split', type=str, default='val', choices=['test', 'val'],
                        help='specify the split of data: val | test')
    parser.add_argument('-b', '--bbox', action='store_true',
                        help='specify to generate ply with bounding box or colored object')
    parser.set_defaults(bbox=False)
    parser.add_argument('-m', '--mode', type=str, default='semantic', choices=['semantic', 'instance'],
                        help='specify instance or semantic mode: semantic | instance')
    parser.add_argument('-o', '--output_dir', type=str, default='output_ply',
                        help='Spiciy the directory of the output ply')
    parser.add_argument('--nms', action='store_true',
                        help='choose to run non_maximum_suppression or not')
    parser.set_defaults(nms=False)
    args = parser.parse_args()
    args.rgb_file_dir = os.path.join(Path(os.getcwd()).parent.parent.absolute(), 'data/scannet', args.split)
    if args.bbox == True:
        args.output_dir = os.path.join(args.output_dir, "bbox")
    else:
        args.output_dir = os.path.join(args.output_dir, "color")
    args.output_dir = os.path.join(args.output_dir, args.mode)
    initialColor()
    generate_pred_inst_ply(args)
