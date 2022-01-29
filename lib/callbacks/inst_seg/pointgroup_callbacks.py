import os
import h5py
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from lib.utils.eval import get_nms_instances
from lib.utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from lib.utils.bbox import get_3d_box_batch


class GTFeaturesGenCallback(Callback):
    
    def __init__(self, ):
        super(GTFeaturesGenCallback, self).__init__()
    
    
    def on_validation_epoch_start(self, trainer, pl_module):
        self.cfg = pl_module.cfg
        self.database = h5py.File(os.path.join(self.cfg.general.root, f"{self.cfg.data.split}.gt_feats.hdf5"), "w", libver="latest")
        self.epoch_dict = {}
    
    
    def on_validation_batch_end(self, trainer, pl_module, data_dict, batch, batch_idx, dataloader_idx):
        scene_id = data_dict["scene_id"][0]
        features = data_dict['proposal_feats']
        bbox_corners = data_dict["bbox_corner"][0]
        object_ids = data_dict["object_ids"][0]
        transformation = data_dict["transformation"][0]
        
        self.epoch_dict[scene_id] = {"object_ids": [],
                                    "features": [],
                                    "bbox_corners": [],
                                    "transformation": []
                                }
        
        for inst_i in range(len(features)):
            cur_feat = features[inst_i]
            cur_corners = bbox_corners[inst_i]
            object_id = object_ids[inst_i]
            
            self.epoch_dict[scene_id]["object_ids"].append(object_id.cpu().numpy())
            self.epoch_dict[scene_id]["features"].append(cur_feat.cpu().numpy())
            self.epoch_dict[scene_id]["bbox_corners"].append(cur_corners.cpu().numpy())
            self.epoch_dict[scene_id]["transformation"].append(transformation.cpu().numpy())
                
                
    def on_validation_epoch_end(self, trainer, pl_module):
        for scene_id in epoch_dict:
            # save scene object ids
            object_id_dataset = "{}|{}_gt_ids".format(str(e), scene_id)
            object_ids = np.array(epoch_dict[scene_id]["object_ids"])
            self.database.create_dataset(object_id_dataset, data=object_ids)

            # save features
            feature_dataset = "{}|{}_features".format(str(e), scene_id)
            features = np.stack(epoch_dict[scene_id]["features"], axis=0)
            self.database.create_dataset(feature_dataset, data=features)

            # save bboxes
            bbox_dataset = "{}|{}_bbox_corners".format(str(e), scene_id)
            bbox_corners = np.stack(epoch_dict[scene_id]["bbox_corners"], axis=0)
            self.database.create_dataset(bbox_dataset, data=bbox_corners)
            
            # save GT bboxes
            gt_dataset = "{}|{}_gt_corners".format(str(e), scene_id)
            gt_corners = np.stack(epoch_dict[scene_id]["bbox_corners"], axis=0)
            self.database.create_dataset(gt_dataset, data=gt_corners)
            
            # save transformations
            trans_dataset = "{}|{}_transformation".format(str(e), scene_id)
            trans_mat = np.stack(epoch_dict[scene_id]["transformation"], axis=0)
            self.database.create_dataset(trans_dataset, data=trans_mat)
    
    
class ParsePredictionCallback(Callback):
    
    def __init__(self, save_preds=True):
        super(ParsePredictionCallback, self).__init__()
        self.save_preds = save_preds
        
    
    def on_test_start(self, trainer, pl_module):
        self.cfg = pl_module.cfg
        
        self.BBOX_POST_DICT = {
                                "remove_empty_box": False, 
                                "use_3d_nms": True, 
                                "nms_iou": 0.25,
                                "use_old_type_nms": False, 
                                "cls_nms": True, 
                                "per_class_proposal": True,
                                "conf_thresh": 0.09,
                            }
    
    
    def on_test_batch_end(self, trainer, pl_module, data_dict, batch, batch_idx, dataloader_idx):
        ##### parse semantic predictions 
        self.parse_semantic_predictions(data_dict)
        
        ##### parse instance predictions
        if self.current_epoch > self.cfg.cluster.prepare_epochs:
            self.parse_instance_predictions(data_dict)
            self.parse_bbox_predictions(data_dict, self.BBOX_POST_DICT)
    
    
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
            
    
    def parse_bbox_predictions(self, data_dict, config_dict):
        """ Parse predictions to OBB parameters and suppress overlapping boxes
        Args:
            data_dict: dict
            config_dict: dict
                {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
                use_old_type_nms, conf_thresh, per_class_proposal}
        """
        batch_size = len(data_dict["batch_offsets"]) - 1
        
        num_proposal = data_dict["proposal_crop_bboxes"].shape[0]
        pred_bboxes = data_dict["proposal_crop_bboxes"].detach().cpu().numpy() # (nProposals, 9)
        bbox_corners = get_3d_box_batch(pred_bboxes[:, :3], pred_bboxes[:, 3:6], pred_bboxes[:, 6]) # (nProposals, 8, 3) numpy
        pred_sem_cls = pred_bboxes[:, 7] - 2
        pred_sem_cls[pred_sem_cls < 0] = 17
        obj_prob = pred_bboxes[:, 8]

        proposals_batchId = data_dict['proposals_batchId']

        nonempty_box_mask = np.ones((num_proposal,))
        if config_dict['remove_empty_box']:
            # -------------------------------------
            # Remove predicted boxes without any point within them..
            batch_pc = data_dict['point_clouds'].cpu().numpy()[:,:,0:3] # B,N,3
            for i in range(batch_size):
                pc = batch_pc[i,:,:] # (N,3)
                for j in range(K):
                    box3d = bbox_corners[i,j,:,:] # (8,3)
                    # box3d = flip_axis_to_depth(box3d)
                    pc_in_box,inds = extract_pc_in_box3d(pc, box3d)
                    if len(pc_in_box) < 5:
                        nonempty_box_mask[i,j] = 0
            # -------------------------------------

        for b in range(batch_size):
            proposal_batch_idx = torch.nonzero(proposals_batchId == b).view(-1).detach().cpu().numpy()
            num_proposal_batch = len(proposal_batch_idx)
            bbox_corners_batch = bbox_corners[proposal_batch_idx]
            obj_prob_batch = obj_prob[proposal_batch_idx]
            pred_sem_cls_batch = pred_sem_cls[proposal_batch_idx]
            pred_mask = np.zeros((num_proposal_batch,))
            if not config_dict['use_3d_nms']:
                # ---------- NMS input: pred_with_prob in (B,K,7) -----------
                boxes_2d_with_prob = np.zeros((num_proposal_batch, 5))
                for j in range(num_proposal_batch):
                    boxes_2d_with_prob[j,0] = np.min(bbox_corners_batch[j,:,0])
                    boxes_2d_with_prob[j,2] = np.max(bbox_corners_batch[j,:,0])
                    boxes_2d_with_prob[j,1] = np.min(bbox_corners_batch[j,:,2])
                    boxes_2d_with_prob[j,3] = np.max(bbox_corners_batch[j,:,2])
                    boxes_2d_with_prob[j,4] = obj_prob_batch[j]
                nonempty_box_inds = np.where(nonempty_box_mask[proposal_batch_idx]==1)[0]
                pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_inds,:], config_dict['nms_iou'], config_dict['use_old_type_nms'])
                assert(len(pick)>0)
                pred_mask[nonempty_box_inds[pick]] = 1
                # ---------- NMS output: pred_mask in (B,K) -----------
            elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
                # ---------- NMS input: pred_with_prob in (B,K,7) -----------
                boxes_3d_with_prob = np.zeros((num_proposal_batch, 7))
                for j in range(num_proposal_batch):
                    boxes_3d_with_prob[j,0] = np.min(bbox_corners_batch[j,:,0])
                    boxes_3d_with_prob[j,1] = np.min(bbox_corners_batch[j,:,1])
                    boxes_3d_with_prob[j,2] = np.min(bbox_corners_batch[j,:,2])
                    boxes_3d_with_prob[j,3] = np.max(bbox_corners_batch[j,:,0])
                    boxes_3d_with_prob[j,4] = np.max(bbox_corners_batch[j,:,1])
                    boxes_3d_with_prob[j,5] = np.max(bbox_corners_batch[j,:,2])
                    boxes_3d_with_prob[j,6] = obj_prob_batch[j]
                nonempty_box_inds = np.where(nonempty_box_mask[proposal_batch_idx]==1)[0]
                pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_inds,:], config_dict['nms_iou'], config_dict['use_old_type_nms'])
                assert(len(pick)>0)
                pred_mask[nonempty_box_inds[pick]] = 1
                # ---------- NMS output: pred_mask in (B,K) -----------
            elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
                # ---------- NMS input: pred_with_prob in (B,K,8) -----------
                boxes_3d_with_prob = np.zeros((num_proposal_batch, 8))
                for j in range(num_proposal_batch):
                    boxes_3d_with_prob[j,0] = np.min(bbox_corners_batch[j,:,0])
                    boxes_3d_with_prob[j,1] = np.min(bbox_corners_batch[j,:,1])
                    boxes_3d_with_prob[j,2] = np.min(bbox_corners_batch[j,:,2])
                    boxes_3d_with_prob[j,3] = np.max(bbox_corners_batch[j,:,0])
                    boxes_3d_with_prob[j,4] = np.max(bbox_corners_batch[j,:,1])
                    boxes_3d_with_prob[j,5] = np.max(bbox_corners_batch[j,:,2])
                    boxes_3d_with_prob[j,6] = obj_prob_batch[j]
                    boxes_3d_with_prob[j,7] = pred_sem_cls_batch[j] # only suppress if the two boxes are of the same class!!
                nonempty_box_inds = np.where(nonempty_box_mask[proposal_batch_idx]==1)[0]
                pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_inds,:], config_dict['nms_iou'], config_dict['use_old_type_nms'])
                assert(len(pick)>0)
                pred_mask[nonempty_box_inds[pick]] = 1
                # ---------- NMS output: pred_mask in (B,K) -----------
                
            bbox_corners_batch = bbox_corners_batch[pred_mask == 1]
            pred_sem_cls_batch = pred_sem_cls_batch[pred_mask == 1]
            obj_prob_batch = obj_prob_batch[pred_mask == 1]
            
            if self.save_preds:
                scene_id = data_dict["scene_id"][0]
                pred_path = os.path.join(self.cfg.general.root, self.cfg.data.split)
                bbox_path = os.path.join(pred_path, "detection")
                os.makedirs(bbox_path, exist_ok=True)
                torch.save({"pred_bbox": bbox_corners_batch, "pred_sem_cls": pred_sem_cls_batch, "pred_obj_prob": obj_prob_batch, "gt_bbox": data_dict['gt_bbox'][b].detach().cpu().numpy(), "gt_bbox_label": data_dict["gt_bbox_label"][b].detach().cpu().numpy(), "gt_sem_cls": data_dict["sem_cls_label"][b].detach().cpu().numpy()}, os.path.join(bbox_path, f"{scene_id}.pth"))
                