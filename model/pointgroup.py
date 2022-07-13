import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from lib.evaluation.instance_segmentation import ScanNetEval, get_gt_instances
from lib.evaluation.object_detection import evaluate_bbox_acc, get_gt_bbox
from lib.common_ops.functions import pointgroup_ops
from lib.common_ops.functions import common_ops
from lib.loss import *
from model.helper import clusters_voxelization, get_batch_offsets
from lib.loss.utils import get_segmented_scores
from lib.util.eval import get_nms_instances
from model.module import Backbone, TinyUnet
from lib.optimizer import init_optimizer, cosine_lr_decay
from lib.evaluation.semantic_segmentation import *


class PointGroup(pl.LightningModule):
    def __init__(self, model, data, optimizer, lr_decay):
        super().__init__()
        self.save_hyperparameters()

        input_channel = model.use_coord * 3 + model.use_color * 3 + model.use_normal * 3
        output_channel = model.m
        semantic_classes = data.classes
        self.instance_classes = semantic_classes - len(data.ignore_classes)

        """
            Backbone Block
        """
        self.backbone = Backbone(input_channel=input_channel,
                                 output_channel=model.m,
                                 block_channels=model.blocks,
                                 block_reps=model.block_reps,
                                 sem_classes=semantic_classes)

        """
            ScoreNet Block
        """
        self.score_net = TinyUnet(output_channel)
        self.score_branch = nn.Linear(output_channel, 1)

    def forward(self, data_dict):
        output_dict = {}

        backbone_output_dict = self.backbone(data_dict["voxel_feats"], data_dict["voxel_locs"], data_dict["v2p_map"])
        output_dict.update(backbone_output_dict)

        if self.current_epoch > self.hparams.model.prepare_epochs or self.hparams.model.freeze_backbone:
            # get prooposal clusters
            batch_idxs = data_dict["vert_batch_ids"].int()
            semantic_preds = output_dict["semantic_scores"].max(1)[1]

            # set mask
            semantic_preds_mask = torch.ones_like(semantic_preds, dtype=torch.bool)
            for class_label in self.hparams.data.ignore_classes:
                semantic_preds_mask = semantic_preds_mask & (semantic_preds != class_label)
            object_idxs = torch.nonzero(semantic_preds_mask).view(-1)  # exclude predicted wall and floor

            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = get_batch_offsets(batch_idxs_, self.hparams.data.batch_size, self.device)
            coords_ = data_dict["locs"][object_idxs]
            pt_offsets_ = output_dict["point_offsets"][object_idxs]

            semantic_preds_cpu = semantic_preds[object_idxs].cpu().int()
            object_idxs_cpu = object_idxs.cpu()

            idx_shift, start_len_shift = common_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, self.hparams.model.cluster.cluster_radius, self.hparams.model.cluster.cluster_shift_meanActive)
            proposals_idx_shift, proposals_offset_shift = pointgroup_ops.pg_bfs_cluster(semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.hparams.model.cluster.cluster_npoint_thre)
            proposals_idx_shift[:, 1] = object_idxs_cpu[proposals_idx_shift[:, 1].long()].int()
            # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset_shift: (nProposal + 1), int
            # proposals_batchId_shift_all: (sumNPoint,) batch id

            idx, start_len = common_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, self.hparams.model.cluster.cluster_radius, self.hparams.model.cluster.cluster_meanActive)
            proposals_idx, proposals_offset = pointgroup_ops.pg_bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(), self.hparams.model.cluster.cluster_npoint_thre)
            proposals_idx[:, 1] = object_idxs_cpu[proposals_idx[:, 1].long()].int()
            # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int

            proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
            proposals_offset_shift += proposals_offset[-1]
            proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
            proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))
            proposals_offset = proposals_offset.cuda()

            # proposals voxelization again
            proposals_voxel_feats, proposals_p2v_map = clusters_voxelization(
                clusters_idx=proposals_idx,
                clusters_offset=proposals_offset,
                feats=output_dict["point_features"],
                coords=data_dict["locs"],
                scale=self.hparams.model.score_scale,
                spatial_shape=self.hparams.model.score_fullscale,
                mode=4,
                device=self.device
            )
            # proposals_voxel_feats: (M, C) M: voxels
            # proposals_p2v_map: point2voxel map (sumNPoint,)
            # score
            score_feats = self.score_net(proposals_voxel_feats)
            pt_score_feats = score_feats.features[proposals_p2v_map.long().cuda()]  # (sumNPoint, C)
            proposals_score_feats = pointgroup_ops.roipool(pt_score_feats, proposals_offset)  # (nProposal, C)
            scores = self.score_branch(proposals_score_feats)  # (nProposal, 1)
            output_dict["proposal_scores"] = (scores, proposals_idx, proposals_offset)
            
        return output_dict

    def configure_optimizers(self):
        return init_optimizer(parameters=self.parameters(), **self.hparams.optimizer)

    def _loss(self, data_dict, output_dict):
        losses = {}

        """semantic loss"""
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda
        sem_seg_criterion = SemSegLoss(self.hparams.data.ignore_label)
        semantic_loss = sem_seg_criterion(output_dict["semantic_scores"], data_dict["sem_labels"])
        losses["semantic_loss"] = semantic_loss

        """offset loss"""
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 12), float32 tensor (meanxyz, center, minxyz, maxxyz)
        # instance_ids: (N), long
        gt_offsets = data_dict["instance_info"] - data_dict["locs"]  # (N, 3)
        valid = data_dict["instance_ids"] != self.hparams.data.ignore_label
        pt_offset_criterion = PTOffsetLoss()
        offset_norm_loss, offset_dir_loss = pt_offset_criterion(output_dict["point_offsets"], gt_offsets, valid_mask=valid)
        losses["offset_norm_loss"] = offset_norm_loss
        losses["offset_dir_loss"] = offset_dir_loss

        total_loss = self.hparams.model.loss_weight[0] * semantic_loss + self.hparams.model.loss_weight[1] * offset_norm_loss + \
                     self.hparams.model.loss_weight[2] * offset_dir_loss

        if self.current_epoch > self.hparams.model.prepare_epochs:
            """score loss"""
            scores, proposals_idx, proposals_offset = output_dict["proposal_scores"]
            instance_pointnum = data_dict["instance_num_point"]
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int
            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset, data_dict["instance_ids"], instance_pointnum) # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(gt_ious, self.hparams.model.fg_thresh, self.hparams.model.bg_thresh)
            score_criterion = ScoreLoss()
            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            losses["score_loss"] = score_loss
            total_loss += self.hparams.model.loss_weight[3] * score_loss
        return losses, total_loss

    def _feed(self, data_dict):
        if self.hparams.model.use_coord:
            data_dict["feats"] = torch.cat((data_dict["feats"], data_dict["locs"]), dim=1)
        data_dict["voxel_feats"] = common_ops.voxelization(data_dict["feats"], data_dict["p2v_map"]) # (M, C), float, cuda
        output_dict = self.forward(data_dict)
        return output_dict

    def training_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self._feed(data_dict)
        losses, total_loss = self._loss(data_dict, output_dict)
        self.log("train/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for key, value in losses.items():
            self.log(f"train/{key}", value, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def training_epoch_end(self, training_step_outputs):
        cosine_lr_decay(self.trainer.optimizers[0], self.hparams.optimizer.lr, self.current_epoch,
                        self.hparams.lr_decay.decay_start_epoch, self.hparams.lr_decay.decay_stop_epoch, 1e-6)


    def validation_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self._feed(data_dict)
        losses, total_loss = self._loss(data_dict, output_dict)

        # log losses
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for key, value in losses.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, sync_dist=True)

        # log semantic prediction accuracy
        semantic_predictions = output_dict["semantic_scores"].max(1)[1].cpu().numpy()
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["sem_labels"].cpu().numpy(),
                                                       ignore_label=self.hparams.data.ignore_label)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["sem_labels"].cpu().numpy(),
                                                   ignore_label=self.hparams.data.ignore_label)
        self.log("val_accuracy/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_accuracy/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True)

        return data_dict, output_dict

    def validation_epoch_end(self, outputs):
        # evaluate instance predictions
        if self.current_epoch > self.hparams.model.prepare_epochs:
            all_pred_insts = []
            all_gt_insts = []
            all_gt_insts_bbox = []
            for batch, output in outputs:
                pred_instances = self._get_pred_instances(batch["scan_ids"][0],
                                                          batch["locs"].cpu().numpy(),
                                                          output["proposal_scores"][0].cpu(),
                                                          output["proposal_scores"][1].cpu(),
                                                          output["proposal_scores"][2].size(0) - 1,
                                                          output["semantic_scores"].cpu())
                gt_instances = get_gt_instances(batch["sem_labels"].cpu(), batch["instance_ids"].cpu(), self.hparams.data.ignore_classes)
                gt_instances_bbox = get_gt_bbox(batch["instance_semantic_cls"].cpu().numpy(),
                                                batch["instance_bboxes"].cpu().numpy(), self.hparams.data.ignore_label)
                all_gt_insts_bbox.append(gt_instances_bbox)
                all_pred_insts.append(pred_instances)
                all_gt_insts.append(gt_instances)
            inst_seg_evaluator = ScanNetEval(self.hparams.data.class_names)
            inst_seg_eval_result = inst_seg_evaluator.evaluate(all_pred_insts, all_gt_insts, print_result=False)

            obj_detect_eval_result = evaluate_bbox_acc(all_pred_insts, all_gt_insts_bbox, self.hparams.data.class_names, print_result=False)

            self.log("val_accuracy/AP", inst_seg_eval_result["all_ap"], sync_dist=True)
            self.log("val_accuracy/AP 50%", inst_seg_eval_result['all_ap_50%'], sync_dist=True)
            self.log("val_accuracy/AP 25%", inst_seg_eval_result["all_ap_25%"], sync_dist=True)
            self.log("val_accuracy/Bounding Box AP 25%", obj_detect_eval_result["all_bbox_ap_0.25"]["avg"],
                     sync_dist=True)
            self.log("val_accuracy/Bounding Box AP 50%", obj_detect_eval_result["all_bbox_ap_0.5"]["avg"],
                     sync_dist=True)

    def test_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self._feed(data_dict)
        return output_dict

    def predict_step(self, data_dict, batch_idx, dataloader_idx=0):
        # prepare input and forward
        output_dict = self._feed(data_dict)
        return data_dict, output_dict

    def test_epoch_end(self, results):
        # evaluate instance predictions
        if self.current_epoch > self.hparams.model.prepare_epochs:
            all_pred_insts = []
            all_gt_insts = []
            all_gt_insts_bbox = []
            all_sem_acc = []
            all_sem_miou = []
            for batch, output in results:
                pred_instances = self._get_pred_instances(batch["scan_ids"][0],
                                                          batch["locs"].cpu().numpy(),
                                                          output["proposal_scores"][0].cpu(),
                                                          output["proposal_scores"][1].cpu(),
                                                          output["proposal_scores"][2].size(0) - 1,
                                                          output["semantic_scores"].cpu())
                gt_instances = get_gt_instances(batch["sem_labels"].cpu(), batch["instance_ids"].cpu(),
                                                self.hparams.data.ignore_classes)
                gt_instances_bbox = get_gt_bbox(batch["instance_semantic_cls"].cpu().numpy(),
                                                batch["instance_bboxes"].cpu().numpy(), self.hparams.data.ignore_label)
                all_gt_insts_bbox.append(gt_instances_bbox)
                all_pred_insts.append(pred_instances)
                all_gt_insts.append(gt_instances)

            if self.hparams.inference.save_predictions:
                inst_pred_path = os.path.join(self.hparams.inference.output_dir, "instance")
                inst_pred_masks_path = os.path.join(inst_pred_path, "predicted_masks")
                os.makedirs(inst_pred_masks_path, exist_ok=True)
                scan_instance_count = {}

                for preds in all_pred_insts:
                    for pred in preds:
                        scan_id = pred["scan_id"]
                        if scan_id not in scan_instance_count:
                            scan_instance_count[scan_id] = 0
                        with open(os.path.join(inst_pred_path, f"{scan_id}.txt"), "a") as f:
                            f.write(
                                f"predicted_masks/{scan_id}_{scan_instance_count[scan_id]:03d}.txt {pred['label_id']} {pred['conf']:.4f}\n")
                        np.savetxt(
                            os.path.join(inst_pred_masks_path, f"{scan_id}_{scan_instance_count[scan_id]:03d}.txt"),
                            pred["pred_mask"], fmt="%d")
                        scan_instance_count[scan_id] += 1
                self.print(f"\nPredictions saved at {inst_pred_path}\n")

            if self.hparams.inference.evaluate:
                inst_seg_evaluator = ScanNetEval(self.hparams.data.class_names)
                self.print("==> Evaluating instance segmentation ...")
                inst_seg_eval_result = inst_seg_evaluator.evaluate(all_pred_insts, all_gt_insts, print_result=True)
                obj_detect_eval_result = evaluate_bbox_acc(all_pred_insts, all_gt_insts_bbox, self.hparams.data.class_names, print_result=True)

                sem_miou_avg = np.mean(np.array(all_sem_miou))
                sem_acc_avg = np.mean(np.array(all_sem_acc))
                self.print(f"Semantic Accuracy: {sem_acc_avg}")
                self.print(f"Semantic mean IoU: {sem_miou_avg}")

    def _get_pred_instances(self, scan_id, gt_xyz, proposals_scores, proposals_idx, num_proposals, semantic_scores):
        semantic_pred_labels = semantic_scores.max(1)[1]
        proposals_score = torch.sigmoid(proposals_scores.view(-1))  # (nProposal,) float
        # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        # proposals_offset: (nProposal + 1), int, cpu

        N = semantic_scores.shape[0]

        proposals_mask = torch.zeros((num_proposals, N), dtype=torch.bool, device="cpu")  # (nProposal, N), int, cuda
        proposals_mask[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = True

        # score threshold & min_npoint mask
        proposals_npoint = proposals_mask.sum(1)
        proposals_thres_mask = torch.logical_and(proposals_score > self.hparams.model.test.TEST_SCORE_THRESH,
                                                 proposals_npoint > self.hparams.model.test.TEST_NPOINT_THRESH)

        proposals_score = proposals_score[proposals_thres_mask]
        proposals_mask = proposals_mask[proposals_thres_mask]

        # instance masks non_max_suppression
        if proposals_score.shape[0] == 0:
            pick_idxs = np.empty(0)
        else:
            proposals_mask_f = proposals_mask.float()  # (nProposal, N), float
            intersection = torch.mm(proposals_mask_f, proposals_mask_f.t())  # (nProposal, nProposal), float
            proposals_npoint = proposals_mask_f.sum(1)  # (nProposal), float, cuda
            proposals_np_repeat_h = proposals_npoint.unsqueeze(-1).repeat(1, proposals_npoint.shape[0])
            proposals_np_repeat_v = proposals_npoint.unsqueeze(0).repeat(proposals_npoint.shape[0], 1)
            cross_ious = intersection / (
                    proposals_np_repeat_h + proposals_np_repeat_v - intersection)  # (nProposal, nProposal), float, cuda
            pick_idxs = get_nms_instances(cross_ious.numpy(), proposals_score.numpy(),
                                          self.hparams.model.test.TEST_NMS_THRESH)  # int, (nCluster,)

        clusters_mask = proposals_mask[pick_idxs].numpy()  # int, (nCluster, N)
        score_pred = proposals_score[pick_idxs].numpy()  # float, (nCluster,)
        nclusters = clusters_mask.shape[0]
        instances = []
        for i in range(nclusters):
            cluster_i = clusters_mask[i]  # (N)
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = semantic_pred_labels[cluster_i][0].item() - 1
            pred['conf'] = score_pred[i]
            pred['pred_mask'] =cluster_i
            pred_inst = gt_xyz[cluster_i]
            pred['pred_bbox'] = np.concatenate((pred_inst.min(0), pred_inst.max(0)))
            instances.append(pred)
        return instances
