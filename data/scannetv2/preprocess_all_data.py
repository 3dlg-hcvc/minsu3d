"""
REFERENCE TO https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
"""

import os
import csv
import json
import torch
import hydra
import numpy as np
import open3d as o3d
from functools import partial
from tqdm.contrib.concurrent import process_map


def get_semantic_mapping_file(file_path):
    label_mapping = {}
    with open(file_path, "r") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        next(tsv_file)  # skip the header
        for line in tsv_file:
            label_mapping[line[1]] = int(line[4])  # use nyu40 label set
    return label_mapping


def read_mesh_file(mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    return np.asarray(mesh.vertices, dtype=np.float32), \
           np.rint(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8), \
           np.asarray(mesh.vertex_normals, dtype=np.float32)


def get_semantic_labels(obj_name_to_segs, seg_to_verts, num_verts, label_map, filtered_label_map):
    semantic_labels = np.full(shape=num_verts, fill_value=-1, dtype=np.int16)
    for label, segs in obj_name_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            if label not in label_map or label_map[label] not in filtered_label_map:
                if label == "ceiling":
                    semantic_label = -1
                else:
                    semantic_label = 19
            else:
                semantic_label = filtered_label_map[label_map[label]]
            semantic_labels[verts] = semantic_label
    return semantic_labels


def read_agg_file(file_path):
    object_id_to_segs = {}
    obj_name_to_segs = {}
    with open(file_path, "r") as f:
        data = json.load(f)
    for group in data['segGroups']:
        object_name = group['label']
        segments = group['segments']
        object_id_to_segs[group["objectId"]] = segments
        if object_name in obj_name_to_segs:
            obj_name_to_segs[object_name].extend(segments)
        else:
            obj_name_to_segs[object_name] = segments.copy()
    return object_id_to_segs, obj_name_to_segs


def read_seg_file(seg_file):
    seg2verts = {}
    with open(seg_file, 'r') as json_data:
        data = json.load(json_data)
    for vert, seg in enumerate(data['segIndices']):
        if seg not in seg2verts:
            seg2verts[seg] = []
        seg2verts[seg].append(vert)
    return seg2verts


def get_instance_ids(object_id2segs, seg2verts, sem_labels, invalid_ids):
    object_id2label_id = {}
    instance_ids = np.full(shape=len(sem_labels), fill_value=-1, dtype=np.int16)
    new_object_id = 0
    for _, segs in object_id2segs.items():
        for seg in segs:
            verts = seg2verts[seg]
            if sem_labels[verts][0] in invalid_ids:
                # skip room architecture and point with invalid semantic labels
                new_object_id -= 1
                break
            instance_ids[verts] = new_object_id
        new_object_id += 1
    return instance_ids


def process_one_scan(scan, cfg, split, label_map):
    mesh_file_path = os.path.join(cfg.data.raw_scene_path, scan, scan + '_vh_clean_2.ply')
    agg_file_path = os.path.join(cfg.data.raw_scene_path, scan, scan + '.aggregation.json')
    seg_file_path = os.path.join(cfg.data.raw_scene_path, scan, scan + '_vh_clean_2.0.010000.segs.json')

    # read mesh_file
    xyz, rgb, normal = read_mesh_file(mesh_file_path)
    num_verts = len(xyz)

    if os.path.exists(agg_file_path):
        # read seg_file
        seg2verts = read_seg_file(seg_file_path)
        # read agg_file
        object_id2segs, label2segs = read_agg_file(agg_file_path)

        # get semantic labels
        # create a map, skip invalid labels to make the final semantic labels consecutive
        filtered_label_map = {}
        invalid_ids = []
        for i, sem_id in enumerate(cfg.data.mapping_classes_ids):
            filtered_label_map[sem_id] = i
            if sem_id in cfg.data.ignore_classes:
                invalid_ids.append(filtered_label_map[sem_id])
        sem_labels = get_semantic_labels(label2segs, seg2verts, num_verts, label_map, filtered_label_map)
        # get instance labels
        instance_ids = get_instance_ids(object_id2segs, seg2verts, sem_labels, invalid_ids)
    else:
        # use zero as placeholders for the test scene
        sem_labels = np.full(shape=num_verts, fill_value=-1, dtype=np.int16)
        instance_ids = np.full(shape=num_verts, fill_value=-1, dtype=np.int16)
    torch.save({'xyz': xyz, 'rgb': rgb, 'normal': normal, 'sem_labels': sem_labels, 'instance_ids': instance_ids},
               os.path.join(cfg.data.dataset_path, split, f"{scan}.pth"))


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    label_map = get_semantic_mapping_file(cfg.data.metadata.combine_file)
    print("\nDefault: using all CPU cores.")
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(cfg.data.dataset_path, split), exist_ok=True)
        with open(getattr(cfg.data.metadata, f"{split}_list")) as f:
            id_list = [line.strip() for line in f]
        print(f"==> Processing {split} split ...")
        process_map(partial(process_one_scan, cfg=cfg, split=split, label_map=label_map), id_list, chunksize=1, max_workers=1)


if __name__ == '__main__':
    main()
