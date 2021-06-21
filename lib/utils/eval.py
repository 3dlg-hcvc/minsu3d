import os
import numpy as np


def write_gt_sem_inst_ids(sem_labels, instance_ids, file_path):
    """ Generate instance txt files for evaluation. Each line represents a number xx00y combining semantic label (x) and instance id (y) for each point.

    Args:
        sem_labels (np.array): {0,1,...,20} (N,) 0:unannotated
        instance_ids (np.array): {0,1,...,instance_num} (N,) 0:unannotated
    """
    from data.nyuv2.model_utils import NYU20_CLASS_IDX
    # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)
    sem_class_idx = np.array(NYU20_CLASS_IDX, dtype=np.int)[sem_labels]
    sem_inst_encoding = sem_class_idx * 1000 + instance_ids # np.zeros(instance_ids.shape, dtype=np.int32)  
    # instance_num = int(instance_ids.max())
    # for inst_id in range(1, instance_num+1):
    #     instance_mask = np.where(instance_ids == inst_id)[0]
    #     sem_seg = sem_labels[instance_mask]
    #     unique_sem_ids, sem_id_counts = np.unique(sem_seg, return_counts=True)
    #     sem_id = unique_sem_ids[np.argmax(sem_id_counts)] # choose the most frequent semantic id
        
    #     semantic_label = NYU20_CLASS_IDX[sem_id]
    #     sem_inst_encoding[instance_mask] = semantic_label * 1000 + inst_id
    np.savetxt(file_path, sem_inst_encoding, fmt='%d')
    

def read_sem_ids(file_path, mode):
    with open(file_path, 'r') as f:
        if mode == 'gt':
            sem_class_idx = [int(encoded_id.rstrip()) // 1000 for encoded_id in f.readlines()]
        else:
            sem_class_idx = [int(encoded_id.rstrip()) for encoded_id in f.readlines()]
        sem_class_idx = np.array(sem_class_idx, dtype=np.int)
    return sem_class_idx


def read_inst_ids(file_path, mode):
    with open(file_path, 'r') as f:
        instance_ids = [int(encoded_id.rstrip()) for encoded_id in f.readlines()]
        instance_ids = np.array(instance_ids, dtype=np.int)
        if mode == 'gt':
            sem_class_idx, instance_ids = instance_ids // 1000, instance_ids % 1000
        else:
            sem_class_idx = None
    return sem_class_idx, instance_ids


def parse_inst_pred_file(file_path, alignment=None):
    lines = open(file_path).read().splitlines()
    instance_info = {}
    for line in lines:
        info = {}
        if alignment:
            mask_rel_path, class_idx, aligned_token_idx, confidence = line.split(' ')
            info['aligned_token_idx'] = int(aligned_token_idx)
        else:
            mask_rel_path, class_idx, confidence = line.split(' ')
        mask_file = os.path.join(os.path.dirname(file_path), mask_rel_path)
        info["class_idx"] = int(class_idx)
        info["confidence"] = float(confidence)
        instance_info[mask_file] = info
    return instance_info


def get_nms_instances(cross_ious, scores, threshold):
    """ non max suppression for 3D instance proposals based on cross ious and scores

    Args:
        ious (np.array): cross ious, (n, n)
        scores (np.array): scores for each proposal, (n,)
        threshold (float): iou threshold

    Returns:
        np.array: idx of picked instance proposals
    """
    # ixs = scores.argsort()[::-1]
    ixs = np.argsort(-scores) # descending order
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        ious = cross_ious[i, ixs[1:]]
        remove_ixs = np.where(ious > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
        
    return np.array(pick, dtype=np.int32)


class Instance(object):
    target_id = 0
    class_idx = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, instance_ids, class_idx, target_id, aligned_token_idx=[]):
        self.target_id = int(target_id)
        self.class_idx = int(class_idx)
        self.aligned_token_idx = aligned_token_idx
        self.vert_count = int(self.get_num_verts(instance_ids, target_id))

    def get_num_verts(self, instance_ids, target_id):
        return (instance_ids == target_id).sum()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict["target_id"] = self.target_id
        dict["class_idx"] = self.class_idx
        dict["aligned_token_idx"] = self.aligned_token_idx
        dict["vert_count"] = self.vert_count
        dict["med_dist"] = self.med_dist
        dict["dist_conf"] = self.dist_conf
        return dict

    def from_json(self, data):
        self.target_id = int(data["target_id"])
        self.class_idx = int(data["class_idx"])
        self.aligned_token_idx = dict["aligned_token_idx"]
        self.vert_count = int(data["vert_count"])
        if ("med_dist" in data):
            self.med_dist = float(data["med_dist"])
            self.dist_conf = float(data["dist_conf"])

    def __str__(self):
        return f"({str(self.target_id)})"


def get_instances(sem_class_idx, instance_ids, gt_class_idx, gt_class_names, id2name):
    instances = {}
    for class_name in gt_class_names:
        instances[class_name] = []
    unique_inst_ids = np.unique(instance_ids)
    for inst_id in unique_inst_ids:
        if inst_id == 0:
            continue
        class_idx = np.argmax(np.bincount(sem_class_idx[instance_ids == inst_id]))
        inst = Instance(instance_ids, class_idx, inst_id)
        if inst.class_idx in gt_class_idx:
            instances[id2name[inst.class_idx]].append(inst.to_dict())
    return instances
