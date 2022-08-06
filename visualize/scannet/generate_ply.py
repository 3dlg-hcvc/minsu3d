import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

from generate_one_ply import generate_single_ply


def generate_pred_inst_ply(args):
    metadata_path = os.path.join(Path(args.rgb_file_dir).parent.parent.absolute(), 'meta_data')
    scene_ids_file = os.path.join(metadata_path, f'scannetv2_{args.split}.txt')
    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    for scene_id in tqdm(scene_ids):
        args.scene_id = scene_id
        generate_single_ply(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predict_dir', type=str, default='../../output/ScanNet/SoftGroup/test/predictions/instance', help='Spiciy the directory of the predictions')
    parser.add_argument('-r', '--rgb_file_dir', type=str, default='../../data/scannet/splited_data')
    parser.add_argument('-s', '--split', type=str, default='val', choices=['test', 'val'],help='specify the split of data: val | test')
    parser.add_argument('-m', '--mode', type=str, default='instance', choices=['semantic', 'instance'],help='specify instance or semantic mode: semantic | instance | detection')
    parser.add_argument('-t', '--type', type=str, default='pointcloud', help='specify type of ply: pointcloud | mesh')
    parser.add_argument('-o', '--output_dir', type=str, default='output_ply', help='Spiciy the directory of the output ply')
    args = parser.parse_args()

    args.rgb_file_dir = os.path.join(args.rgb_file_dir, args.split)
    args.output_dir = os.path.join(args.output_dir, args.mode)
    generate_pred_inst_ply(args)
