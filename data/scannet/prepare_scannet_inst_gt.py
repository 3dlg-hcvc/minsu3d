'''
Generate instance groundtruth .txt files (for evaluation)
'''

import numpy as np
import glob
import torch
import os

semantic_label_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
semantic_label_names = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']


if __name__ == '__main__':
    split = 'val'
    files = sorted(glob.glob(f'splited_data/{split}/scene*.pth'))
    rooms = [torch.load(i) for i in files]

    os.makedirs(f'splited_gt/{split}', exist_ok=True)

    for i in range(len(rooms)):
        instance_ids = rooms[i]["instance_ids"]
        label = rooms[i]["sem_labels"]  # {-1,0,1,...,19}

        # xyz, rgb, label, instance_ids = rooms[i]   # label 0~19 -1;  instance_ids 0~instance_num-1 -1
        scene_name = files[i].split('/')[-1][:12]
        print('{}/{} {}'.format(i + 1, len(rooms), scene_name))

        # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)
        # instance_ids_new = np.zeros(instance_ids.shape, dtype=np.int32)
        instance_ids_new = (label.copy().astype(np.int32) + 1) * 1000

        instance_num = int(instance_ids.max()) + 1
        for inst_id in range(instance_num):
            instance_mask = np.where(instance_ids == inst_id)[0]
            sem_id = int(label[instance_mask[0]])
            if(sem_id == -1):
                semantic_label = 0
            else:
                semantic_label = semantic_label_idxs[sem_id]
            instance_ids_new[instance_mask] = semantic_label * 1000 + inst_id + 1
            # instance_ids_new[instance_mask] += inst_id + 1

        np.savetxt(f'splited_gt/{split}/{scene_name}.txt', instance_ids_new, fmt='%d')





