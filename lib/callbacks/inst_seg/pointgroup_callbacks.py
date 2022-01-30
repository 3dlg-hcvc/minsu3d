import os
import h5py
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from lib.utils.eval import get_nms_instances


class ParsePredictionCallback(Callback):
    
    def __init__(self, save_preds=True):
        super(ParsePredictionCallback, self).__init__()
        self.save_preds = save_preds
        
    
    def on_test_start(self, trainer, pl_module):
        self.cfg = pl_module.cfg
    
    
    def on_test_batch_end(self, trainer, pl_module, data_dict, batch, batch_idx, dataloader_idx):
        ##### parse semantic predictions 
        self.parse_semantic_predictions(data_dict)
        
        ##### parse instance predictions
        if self.current_epoch > self.cfg.cluster.prepare_epochs:
            self.parse_instance_predictions(data_dict)
    
    
    def parse_semantic_predictions(self, data_dict):
        from data.scannet.model_util_scannet import NYU20_CLASS_IDX
        NYU20_CLASS_IDX = NYU20_CLASS_IDX[1:] # for scannet
        
        ##### (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
        semantic_scores = data_dict["semantic_scores"]  # (N, nClass) float32, cuda
        semantic_pred_labels = semantic_scores.max(1)[1]  # (N) long, cuda
        semantic_class_idx = torch.tensor(NYU20_CLASS_IDX, dtype=torch.int).cuda() # (nClass)
        semantic_pred_class_idx = semantic_class_idx[semantic_pred_labels].cpu().numpy()
        data_dict["semantic_pred_class_idx"] = semantic_pred_class_idx
        
        ##### save predictions
        if self.save_preds:
            scene_id = data_dict["scene_id"][0]
            pred_path = os.path.join(self.cfg.general.root, self.cfg.data.split)
            sem_pred_path = os.path.join(pred_path, "semantic")
            os.makedirs(sem_pred_path, exist_ok=True)
            sem_pred_file_path = os.path.join(sem_pred_path, f"{scene_id}.txt")
            np.savetxt(sem_pred_file_path, semantic_pred_class_idx, fmt="%d")


    def parse_semantic_predictions(self, data_dict):
        from data.scannet.model_util_scannet import NYU20_CLASS_IDX
        NYU20_CLASS_IDX = NYU20_CLASS_IDX[1:] # for scannet
        
        ##### (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
        semantic_scores = data_dict["semantic_scores"]  # (N, nClass) float32, cuda
        semantic_pred_labels = semantic_scores.max(1)[1]  # (N) long, cuda
        semantic_class_idx = torch.tensor(NYU20_CLASS_IDX, dtype=torch.int).cuda() # (nClass)
        semantic_pred_class_idx = semantic_class_idx[semantic_pred_labels].cpu().numpy()
        data_dict["semantic_pred_class_idx"] = semantic_pred_class_idx
        
        ##### save predictions
        if self.save_preds:
            scene_id = data_dict["scene_id"][0]
            pred_path = os.path.join(self.cfg.general.root, self.cfg.data.split)
            sem_pred_path = os.path.join(pred_path, "semantic")
            os.makedirs(sem_pred_path, exist_ok=True)
            sem_pred_file_path = os.path.join(sem_pred_path, f"{scene_id}.txt")
            np.savetxt(sem_pred_file_path, semantic_pred_class_idx, fmt="%d")
            
            
    def parse_instance_predictions(self, data_dict):
        scores, proposals_idx, proposals_offset = data_dict["proposal_scores"]
        proposals_score = torch.sigmoid(scores.view(-1)) # (nProposal,) float, cuda
        # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        # proposals_offset: (nProposal + 1), int, cpu

        num_proposals = proposals_offset.shape[0] - 1
        N = data_dict["semantic_scores"].shape[0]
        
        proposals_mask = torch.zeros((num_proposals, N), dtype=torch.int).cuda() # (nProposal, N), int, cuda
        proposals_mask[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
        
        ##### score threshold & min_npoint mask
        proposals_npoint = data_dict["proposals_npoint"] #proposals_mask.sum(1)
        proposals_thres_mask = data_dict["proposal_thres_mask"] #torch.logical_and(proposals_score > self.cfg.test.TEST_SCORE_THRESH, proposals_npoint > self.cfg.test.TEST_NPOINT_THRESH)
        
        proposals_score = proposals_score[proposals_thres_mask]
        proposals_mask = proposals_mask[proposals_thres_mask]
        
        ##### instance masks non_max_suppression
        if proposals_score.shape[0] == 0:
            pick_idxs = np.empty(0)
        else:
            proposals_mask_f = proposals_mask.float()  # (nProposal, N), float, cuda
            intersection = torch.mm(proposals_mask_f, proposals_mask_f.t())  # (nProposal, nProposal), float, cuda
            proposals_npoint = proposals_mask_f.sum(1)  # (nProposal), float, cuda
            proposals_np_repeat_h = proposals_npoint.unsqueeze(-1).repeat(1, proposals_npoint.shape[0])
            proposals_np_repeat_v = proposals_npoint.unsqueeze(0).repeat(proposals_npoint.shape[0], 1)
            cross_ious = intersection / (proposals_np_repeat_h + proposals_np_repeat_v - intersection) # (nProposal, nProposal), float, cuda
            pick_idxs = get_nms_instances(cross_ious.cpu().numpy(), proposals_score.cpu().numpy(), self.cfg.test.TEST_NMS_THRESH)  # int, (nCluster,)

        clusters_mask = proposals_mask[pick_idxs].cpu().numpy() # int, (nCluster, N)
        clusters_score = proposals_score[pick_idxs].cpu().numpy() # float, (nCluster,)
        nclusters = clusters_mask.shape[0]
        
        if self.save_preds:
            assert "semantic_pred_class_idx" in data_dict, "make sure you parse semantic predictions at first"
            scene_id = data_dict["scene_id"][0]
            semantic_pred_class_idx = data_dict["semantic_pred_class_idx"]
            pred_path = os.path.join(self.cfg.general.root, self.cfg.data.split)
            inst_pred_path = os.path.join(pred_path, "instance")
            inst_pred_masks_path = os.path.join(inst_pred_path, "predicted_masks")
            # os.makedirs(inst_pred_path, exist_ok=True)
            os.makedirs(inst_pred_masks_path, exist_ok=True)
            cluster_ids = np.ones(shape=(N)) * -1 # id starts from 0
            with open(os.path.join(inst_pred_path, f"{scene_id}.txt"), "w") as f:
                for c_id in range(nclusters):
                    cluster_i = clusters_mask[c_id]  # (N)
                    cluster_ids[cluster_i == 1] = c_id
                    assert np.unique(semantic_pred_class_idx[cluster_i == 1]).size == 1
                    cluster_i_class_idx = semantic_pred_class_idx[cluster_i == 1][0]
                    score = clusters_score[c_id]
                    f.write(f"predicted_masks/{scene_id}_{c_id:03d}.txt {cluster_i_class_idx} {score:.4f}\n")
                    np.savetxt(os.path.join(inst_pred_masks_path, f"{scene_id}_{c_id:03d}.txt"), cluster_i, fmt="%d")
            np.savetxt(os.path.join(inst_pred_path, f"{scene_id}.cluster_ids.txt"), cluster_ids, fmt="%d")
                