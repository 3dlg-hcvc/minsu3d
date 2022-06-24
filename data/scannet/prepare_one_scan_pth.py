'''
REFERENCE TO https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
'''

import json, argparse, os
from omegaconf import OmegaConf
import numpy as np
import torch
import scannet_utils
from plyfile import PlyData
import open3d as o3d

IGNORE_CLASS_IDS = np.array([1, 2, 22])  # exclude wall, floor and ceiling

LABEL_MAP_FILE = 'meta_data/scannetv2-labels.combined.tsv'

OBJECT_MAPPING = scannet_utils.get_raw2scannetv2_label_map()

# Map relevant classes to {0,1,...,19}, and ignored classes to -1
remapper = np.full(shape=150, fill_value=-1, dtype=np.int32)
for label, nyu40id in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[nyu40id] = label


# IGNORE_CLASS_IDS = np.array([0, 1]) # wall and floor after remapping


def read_mesh_file(mesh_file, axis_align_matrix):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    if axis_align_matrix is not None:
        # align the mesh
        mesh.transform(axis_align_matrix)
    return np.asarray(mesh.vertices), np.asarray(mesh.vertex_colors), np.asarray(mesh.vertex_normals)


def read_axis_align_matrix(meta_file):
    axis_align_matrix = None
    with open(meta_file, 'r') as f:
        for line in f:
            line_content = line.strip()
            if 'axisAlignment' in line_content:
                axis_align_matrix = [float(x) for x in line_content.strip('axisAlignment = ').split(' ')]
                axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
                break
    return axis_align_matrix


def align_mesh_vertices(mesh, axis_align_matrix):
    aligned_mesh = np.copy(mesh)
    if axis_align_matrix is not None:
        homo_ones = np.ones(shape=(mesh.shape[0], 1))
        aligned_mesh[:, 0:3] = np.dot(np.concatenate([mesh[:, 0:3], homo_ones], 1), axis_align_matrix.transpose())[:,:3]
    return aligned_mesh


def read_label_file(label_file):
    plydata = PlyData.read(label_file)
    sem_labels = np.array(plydata['vertex']['label'], dtype=np.uint32)  # nyu40
    return sem_labels


def read_agg_file(agg_file):
    objectId2segs = {}
    label2segs = {}
    objectId = 0
    with open(agg_file, 'r') as json_data:
        data = json.load(json_data)
        for group in data['segGroups']:
            label = group['label']
            segs = group['segments']
            if OBJECT_MAPPING[label] not in ['wall', 'floor', 'ceiling']:
                objectId2segs[objectId] = segs
                objectId += 1
                if label in label2segs:
                    label2segs[label].extend(segs)
                else:
                    label2segs[label] = segs.copy()
                # objectId += 1
    if agg_file.split('/')[-2] == 'scene0217_00':
        objectIds = sorted(objectId2segs.keys())
        objectId2segs = {objectId: objectId2segs[objectId] for objectId in objectIds[:len(objectId2segs) // 2]}
    return objectId2segs, label2segs


def read_seg_file(seg_file):
    seg2verts = {}
    with open(seg_file, 'r') as json_data:
        data = json.load(json_data)
        num_verts = len(data['segIndices'])
        for vert, seg in enumerate(data['segIndices']):
            if seg in seg2verts:
                seg2verts[seg].append(vert)
            else:
                seg2verts[seg] = [vert]
    return seg2verts, num_verts


def get_instance_ids(objectId2segs, seg2verts, sem_labels):
    objectId2labelId = {}
    # -1: points are not assigned to any objects ( objectId starts from 0)
    instance_ids = np.full(shape=len(sem_labels), fill_value=-1, dtype=np.int32)
    for objectId, segs in objectId2segs.items():
        for seg in segs:
            verts = seg2verts[seg]
            instance_ids[verts] = objectId
        if objectId not in objectId2labelId:
            objectId2labelId[objectId] = sem_labels[verts][0]

        # assert(len(np.unique(sem_labels[pointids])) == 1)
    return instance_ids, objectId2labelId


def get_instance_bboxes(xyz, instance_ids, object_id2label_id):
    num_instances = max(object_id2label_id.keys()) + 1
    instance_bboxes = np.zeros(shape=(num_instances, 8))  # (cx, cy, cz, dx, dy, dz, ins_label, objectId)
    for objectId in object_id2label_id:
        ins_label = object_id2label_id[objectId]  # nyu40id
        obj_pc = xyz[instance_ids == objectId]  #
        if len(obj_pc) == 0:
            continue
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        bbox = np.array(
            [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, xmax - xmin, ymax - ymin, zmax - zmin, ins_label,
             objectId])
        instance_bboxes[objectId, :] = bbox
    return instance_bboxes


def export(scene, cfg):
    mesh_file_path = os.path.join(cfg.SCANNETV2_PATH.raw_scans, scene, scene + '_vh_clean_2.ply')
    label_file_path = os.path.join(cfg.SCANNETV2_PATH.raw_scans, scene, scene + '_vh_clean_2.labels.ply')
    agg_file_path = os.path.join(cfg.SCANNETV2_PATH.raw_scans, scene, scene + '.aggregation.json')
    seg_file_path = os.path.join(cfg.SCANNETV2_PATH.raw_scans, scene, scene + '_vh_clean_2.0.010000.segs.json')
    meta_file_path = os.path.join(cfg.SCANNETV2_PATH.raw_scans, scene, scene + '.txt')

    # read meta_file
    axis_align_matrix = read_axis_align_matrix(meta_file_path)

    # read mesh_file
    xyz, rgb, normal = read_mesh_file(mesh_file_path, axis_align_matrix)
    num_verts = len(xyz)

    if os.path.exists(agg_file_path):
        # read label_file
        sem_labels = read_label_file(label_file_path)
        # read seg_file
        seg2verts, num = read_seg_file(seg_file_path)
        assert num_verts == num
        # read agg_file
        object_id2segs, label2segs = read_agg_file(agg_file_path)
        # get instance labels
        instance_ids, object_id2label_id = get_instance_ids(object_id2segs, seg2verts, sem_labels)
        # get aligned instance bounding boxes
        aligned_instance_bboxes = get_instance_bboxes(xyz, instance_ids, object_id2label_id)
    else:
        # use zero as placeholders for the test scene
        # print("use placeholders")
        sem_labels = np.zeros(shape=num_verts, dtype=np.uint32)  # 0: unannotated
        instance_ids = np.full(shape=num_verts, fill_value=-1, dtype=np.int32)  # -1: unannotated
        aligned_instance_bboxes = np.zeros((1, 8))
    sem_labels = remapper[sem_labels]
    return xyz, rgb, normal, sem_labels, instance_ids, aligned_instance_bboxes


def process_one_scan(scan, cfg):
    xyz, rgb, normal, sem_labels, instance_ids, aligned_instance_bboxes = export(scan, cfg)

    if aligned_instance_bboxes.shape[0] > 1:
        bbox_mask = np.logical_not(np.in1d(aligned_instance_bboxes[:, -2],
                                           IGNORE_CLASS_IDS))  # match the mesh2cap; not care wall, floor and ceiling for instances
        aligned_instance_bboxes = aligned_instance_bboxes[bbox_mask, :]

    torch.save({'xyz': xyz, 'rgb': rgb, 'normal': normal, 'sem_labels': sem_labels, 'instance_ids': instance_ids,
                'aligned_instance_bboxes': aligned_instance_bboxes},
               os.path.join(cfg.SCANNETV2_PATH.splited_data, cfg.split, scan + '.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id', help='scan id', required=True)
    parser.add_argument('-s', '--split', help='data split (train / val / test)', default='train')
    parser.add_argument('-c', '--cfg', help='configuration YAML file', default='../../conf/path.yaml')
    opt = parser.parse_args()
    cfg = OmegaConf.load(opt.cfg)
    cfg.split = opt.split
    os.makedirs(os.path.join(cfg.SCANNETV2_PATH.splited_data, cfg.split), exist_ok=True)
    process_one_scan(opt.id, cfg)
