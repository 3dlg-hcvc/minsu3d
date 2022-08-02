import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from minpg.lib.evaluation.instance_segmentation import GeneralDatasetEvaluator, get_gt_instances, rle_encode, rle_decode
from minpg.lib.evaluation.object_detection import evaluate_bbox_acc, get_gt_bbox
from minpg.lib.common_ops.functions import hais_ops
from minpg.lib.common_ops.functions import common_ops
from minpg.lib.loss import *
from model.helper import clusters_voxelization, get_batch_offsets
from minpg.lib.loss.utils import get_segmented_scores
from model.module import Backbone, TinyUnet
from minpg.lib.optimizer import init_optimizer, cosine_lr_decay
from minpg.lib.evaluation.semantic_segmentation import *
from tqdm import tqdm


class HAIS(pl.LightningModule):
    def __init__(self, model, data, optimizer, lr_decay, inference=None):
        super().__init__()
        self.save_hyperparameters()

        input_channel = model.use_coord * 3 + model.use_color * 3 + model.use_normal * 3 + model.use_multiview * 128
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
            Intra-instance Block
        """
        self.tiny_unet = TinyUnet(output_channel)
        self.score_branch = nn.Linear(output_channel, 1)
        self.mask_branch = nn.Sequential(
            nn.Linear(output_channel, output_channel),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel, 1)
        )

    def forward(self, data_dict, training):
        output_dict = {}

        backbone_output_dict = self.backbone(data_dict["voxel_feats"], data_dict["voxel_locs"], data_dict["v2p_map"])
        output_dict.update(backbone_output_dict)

        if self.current_epoch > self.hparams.model.prepare_epochs:
            # get prooposal clusters
            batch_idxs = data_dict["vert_batch_ids"]
            semantic_preds = output_dict["semantic_scores"].max(1)[1]
            # set mask
            semantic_preds_mask = torch.ones_like(semantic_preds, dtype=torch.bool)
            for class_label in self.hparams.data.ignore_classes:
                semantic_preds_mask = semantic_preds_mask & (semantic_preds != class_label)
            object_idxs = torch.nonzero(semantic_preds_mask).view(-1)  # exclude predicted wall and floor

            batch_idxs_ = batch_idxs[object_idxs].int()
            batch_offsets_ = get_batch_offsets(batch_idxs_, self.hparams.data.batch_size, self.device)
            coords_ = data_dict["locs"][object_idxs]
            pt_offsets_ = output_dict["point_offsets"][object_idxs]

            semantic_preds_cpu = semantic_preds[object_idxs].cpu().int()

            idx_shift, start_len_shift = common_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_,
                                                                      batch_offsets_,
                                                                      self.hparams.model.point_aggr_radius,
                                                                      self.hparams.model.cluster_shift_meanActive)

            using_set_aggr = self.hparams.model.using_set_aggr_in_training if training else self.hparams.model.using_set_aggr_in_testing
            proposals_idx, proposals_offset = hais_ops.hierarchical_aggregation(
                semantic_preds_cpu, (coords_ + pt_offsets_).cpu(), idx_shift.cpu(), start_len_shift.cpu(),
                batch_idxs_.cpu(), using_set_aggr, self.hparams.data.point_num_avg, self.hparams.data.radius_avg,
                self.hparams.data.ignore_label)

            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

            # # restrict the num of training proposals, avoid OOM
            # max_proposal_num = getattr(self.cfg, 'max_proposal_num', 200)
            # if training_mode == 'train' and proposals_offset.shape[0] > max_proposal_num:
            #     proposals_offset = proposals_offset[:max_proposal_num + 1]
            #     proposals_idx = proposals_idx[: proposals_offset[-1]]
            #     assert proposals_idx.shape[0] == proposals_offset[-1]
            #     print('selected proposal num', proposals_offset.shape[0] - 1)

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

            # predict instance scores
            inst_score = self.tiny_unet(proposals_voxel_feats)
            score_feats = inst_score.features[proposals_p2v_map.long()]

            # predict mask scores
            # first linear than voxel to point,  more efficient  (because voxel num < point num)
            mask_scores = self.mask_branch(inst_score.features)[proposals_p2v_map.long()]

            # predict instance scores
            if self.current_epoch > self.hparams.model.use_mask_filter_score_feature_start_epoch:
                mask_index_select = torch.ones_like(mask_scores)
                mask_index_select[torch.sigmoid(mask_scores) < self.hparams.model.mask_filter_score_feature_thre] = 0.
                score_feats = score_feats * mask_index_select
            score_feats = common_ops.roipool(score_feats, proposals_offset.cuda())  # (nProposal, C)
            scores = self.score_branch(score_feats)  # (nProposal, 1)
            output_dict["proposal_scores"] = (scores, proposals_idx, proposals_offset, mask_scores)
        return output_dict

    def configure_optimizers(self):
        return init_optimizer(parameters=self.parameters(), **self.hparams.optimizer)

    def _loss(self, data_dict, output_dict):
        losses = {}

        """semantic loss"""
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda
        sem_seg_criterion = SemSegLoss(self.hparams.data.ignore_label)
        semantic_loss = sem_seg_criterion(output_dict["semantic_scores"], data_dict["sem_labels"].long())
        losses["semantic_loss"] = semantic_loss

        """offset loss"""
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 12), float32 tensor (meanxyz, center, minxyz, maxxyz)
        # instance_ids: (N), long
        gt_offsets = data_dict["instance_info"] - data_dict["locs"]  # (N, 3)
        valid = data_dict["instance_ids"] != self.hparams.data.ignore_label
        pt_offset_criterion = PTOffsetLoss()
        offset_norm_loss, _ = pt_offset_criterion(output_dict["point_offsets"], gt_offsets, valid_mask=valid)
        losses["offset_norm_loss"] = offset_norm_loss

        total_loss = self.hparams.model.loss_weight[0] * semantic_loss + self.hparams.model.loss_weight[
            1] * offset_norm_loss

        if self.current_epoch > self.hparams.model.prepare_epochs:
            """score and mask loss"""
            scores, proposals_idx, proposals_offset, mask_scores = output_dict['proposal_scores']

            # get iou and calculate mask label and mask loss
            mask_scores_sigmoid = torch.sigmoid(mask_scores)

            proposals_idx = proposals_idx[:, 1].cuda()
            proposals_offset = proposals_offset.cuda()

            if self.current_epoch > self.hparams.model.cal_iou_based_on_mask_start_epoch:
                ious = common_ops.get_mask_iou_on_pred(proposals_idx, proposals_offset, data_dict["instance_ids"],
                                                       data_dict["instance_num_point"],
                                                       mask_scores_sigmoid.detach())
            else:
                ious = common_ops.get_mask_iou_on_cluster(proposals_idx, proposals_offset,
                                                          data_dict["instance_ids"],
                                                          data_dict["instance_num_point"])

            mask_label, mask_label_mask = common_ops.get_mask_label(proposals_idx, proposals_offset,
                                                                    data_dict["instance_ids"],
                                                                    data_dict["instance_semantic_cls"],
                                                                    data_dict["instance_num_point"], ious,
                                                                    self.hparams.data.ignore_label, 0.5)
            mask_label = mask_label.unsqueeze(1)
            mask_label_mask = mask_label_mask.unsqueeze(1)
            mask_scoring_criterion = MaskScoringLoss(weight=mask_label_mask, reduction='sum')
            mask_loss = mask_scoring_criterion(mask_scores_sigmoid, mask_label.float())
            mask_loss /= (torch.count_nonzero(mask_label_mask) + 1)
            losses["mask_loss"] = mask_loss

            gt_ious, _ = ious.max(1)  # gt_ious: (nProposal) float, long

            gt_scores = get_segmented_scores(gt_ious, self.hparams.model.fg_thresh, self.hparams.model.bg_thresh)
            score_criterion = ScoreLoss()
            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            losses["score_loss"] = score_loss
            total_loss += self.hparams.model.loss_weight[2] * score_loss + self.hparams.model.loss_weight[3] * mask_loss
        return losses, total_loss

    def _feed(self, data_dict, training):
        if self.hparams.model.use_coord:
            data_dict["feats"] = torch.cat((data_dict["feats"], data_dict["locs"]), dim=1)
        data_dict["voxel_feats"] = common_ops.voxelization(data_dict["feats"],
                                                           data_dict["p2v_map"])  # (M, C), float, cuda
        output_dict = self.forward(data_dict, training)
        return output_dict

    def training_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self._feed(data_dict, True)
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
        output_dict = self._feed(data_dict, True)
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
        self.log("val_eval/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_eval/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True)

        if self.current_epoch > self.hparams.model.prepare_epochs:
            pred_instances = self._get_pred_instances(data_dict["scan_ids"][0],
                                                      data_dict["locs"].cpu().numpy(),
                                                      output_dict["proposal_scores"][0].cpu(),
                                                      output_dict["proposal_scores"][1].cpu(),
                                                      output_dict["proposal_scores"][2].size(0) - 1,
                                                      output_dict["proposal_scores"][3].cpu(),
                                                      output_dict["semantic_scores"].cpu())
            gt_instances = get_gt_instances(data_dict["sem_labels"].cpu(), data_dict["instance_ids"].cpu(),
                                            self.hparams.data.ignore_classes)
            gt_instances_bbox = get_gt_bbox(data_dict["instance_semantic_cls"].cpu().numpy(),
                                            data_dict["instance_bboxes"].cpu().numpy(), self.hparams.data.ignore_label)

            return pred_instances, gt_instances, gt_instances_bbox

    def validation_epoch_end(self, outputs):
        # evaluate instance predictions
        if self.current_epoch > self.hparams.model.prepare_epochs:
            all_pred_insts = []
            all_gt_insts = []
            all_gt_insts_bbox = []
            for pred_instances, gt_instances, gt_instances_bbox in outputs:
                all_gt_insts_bbox.append(gt_instances_bbox)
                all_pred_insts.append(pred_instances)
                all_gt_insts.append(gt_instances)
            inst_seg_evaluator = GeneralDatasetEvaluator(self.hparams.data.class_names, self.hparams.data.ignore_label)
            inst_seg_eval_result = inst_seg_evaluator.evaluate(all_pred_insts, all_gt_insts, print_result=False)

            obj_detect_eval_result = evaluate_bbox_acc(all_pred_insts, all_gt_insts_bbox, self.hparams.data.class_names,
                                                       print_result=False)

            self.log("val_eval/AP", inst_seg_eval_result["all_ap"], sync_dist=True)
            self.log("val_eval/AP 50%", inst_seg_eval_result['all_ap_50%'], sync_dist=True)
            self.log("val_eval/AP 25%", inst_seg_eval_result["all_ap_25%"], sync_dist=True)
            self.log("val_eval/BBox AP 25%", obj_detect_eval_result["all_bbox_ap_0.25"]["avg"],
                     sync_dist=True)
            self.log("val_eval/BBox AP 50%", obj_detect_eval_result["all_bbox_ap_0.5"]["avg"],
                     sync_dist=True)

    def test_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self._feed(data_dict, False)

        sem_labels_cpu = data_dict["sem_labels"].cpu()
        semantic_predictions = output_dict["semantic_scores"].max(1)[1].cpu().numpy()
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions,
                                                       sem_labels_cpu.numpy(),
                                                       ignore_label=self.hparams.data.ignore_label)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, sem_labels_cpu.numpy(),
                                                   ignore_label=self.hparams.data.ignore_label)

        if self.current_epoch > self.hparams.model.prepare_epochs:
            pred_instances = self._get_pred_instances(data_dict["scan_ids"][0],
                                                      data_dict["locs"].cpu().numpy(),
                                                      output_dict["proposal_scores"][0].cpu(),
                                                      output_dict["proposal_scores"][1].cpu(),
                                                      output_dict["proposal_scores"][2].size(0) - 1,
                                                      output_dict["proposal_scores"][3].cpu(),
                                                      output_dict["semantic_scores"].cpu())
            gt_instances = get_gt_instances(sem_labels_cpu, data_dict["instance_ids"].cpu(),
                                            self.hparams.data.ignore_classes)
            gt_instances_bbox = get_gt_bbox(data_dict["instance_semantic_cls"].cpu().numpy(),
                                            data_dict["instance_bboxes"].cpu().numpy(), self.hparams.data.ignore_label)

            return semantic_accuracy, semantic_mean_iou, pred_instances, gt_instances, gt_instances_bbox

    def predict_step(self, data_dict, batch_idx, dataloader_idx=0):
        # prepare input and forward
        output_dict = self._feed(data_dict, False)

    def test_epoch_end(self, results):
        # evaluate instance predictions
        if self.current_epoch > self.hparams.model.prepare_epochs:
            all_pred_insts = []
            all_gt_insts = []
            all_gt_insts_bbox = []
            all_sem_acc = []
            all_sem_miou = []
            for semantic_accuracy, semantic_mean_iou, pred_instances, gt_instances, gt_instances_bbox in results:
                all_sem_acc.append(semantic_accuracy)
                all_sem_miou.append(semantic_mean_iou)
                all_gt_insts_bbox.append(gt_instances_bbox)
                all_pred_insts.append(pred_instances)
                all_gt_insts.append(gt_instances)

            if self.hparams.inference.save_predictions:
                inst_pred_path = os.path.join(self.hparams.inference.output_dir, "instance")
                inst_pred_masks_path = os.path.join(inst_pred_path, "predicted_masks")
                os.makedirs(inst_pred_masks_path, exist_ok=True)
                scan_instance_count = {}

                for preds in tqdm(all_pred_insts, desc="==> Saving predictions ..."):
                    tmp_info = []
                    scan_id = preds[0]["scan_id"]
                    for pred in preds:
                        if scan_id not in scan_instance_count:
                            scan_instance_count[scan_id] = 0
                        tmp_info.append(
                            f"predicted_masks/{scan_id}_{scan_instance_count[scan_id]:03d}.txt {pred['label_id']} {pred['conf']:.4f}\n")
                        np.savetxt(
                            os.path.join(inst_pred_masks_path, f"{scan_id}_{scan_instance_count[scan_id]:03d}.txt"),
                            rle_decode(pred["pred_mask"]), fmt="%d")
                        scan_instance_count[scan_id] += 1
                    with open(os.path.join(inst_pred_path, f"{scan_id}.txt"), "w") as f:
                        for mask_info in tmp_info:
                            f.write(mask_info)
                self.print(f"\nPredictions saved at {inst_pred_path}\n")

            if self.hparams.inference.evaluate:
                inst_seg_evaluator = GeneralDatasetEvaluator(self.hparams.data.class_names,
                                                             self.hparams.data.ignore_label)
                self.print("==> Evaluating instance segmentation ...")
                inst_seg_eval_result = inst_seg_evaluator.evaluate(all_pred_insts, all_gt_insts, print_result=True)
                obj_detect_eval_result = evaluate_bbox_acc(all_pred_insts, all_gt_insts_bbox,
                                                           self.hparams.data.class_names, print_result=True)

                sem_miou_avg = np.mean(np.array(all_sem_miou))
                sem_acc_avg = np.mean(np.array(all_sem_acc))
                self.print(f"Semantic Accuracy: {sem_acc_avg}")
                self.print(f"Semantic mean IoU: {sem_miou_avg}")

    def _get_pred_instances(self, scan_id, gt_xyz, scores, proposals_idx, num_proposals, mask_scores, semantic_scores):
        semantic_pred_labels = semantic_scores.max(1)[1]
        scores_pred = torch.sigmoid(scores.view(-1))

        N = semantic_scores.shape[0]
        # proposals_idx: (sumNPoint, 2), int, cpu, [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N
        # proposals_offset: (nProposal + 1), int, cpu
        proposals_pred = torch.zeros((num_proposals, N), dtype=torch.bool, device="cpu")
        # (nProposal, N), int, cuda

        # outlier filtering
        _mask = mask_scores.squeeze(1) > self.hparams.model.test.test_mask_score_thre
        proposals_pred[proposals_idx[_mask][:, 0].long(), proposals_idx[_mask][:, 1].long()] = True

        # semantic_id = semantic_pred_labels[
        #     proposals_idx[:, 1][proposals_offset[:-1].long()].long()]  # (nProposal), long

        # score threshold
        score_mask = (scores_pred > self.hparams.model.test.TEST_SCORE_THRESH)
        scores_pred = scores_pred[score_mask]
        proposals_pred = proposals_pred[score_mask]
        # semantic_id = semantic_id[score_mask]

        # npoint threshold
        proposals_pointnum = torch.count_nonzero(proposals_pred, dim=1)
        npoint_mask = (proposals_pointnum >= self.hparams.model.test.TEST_NPOINT_THRESH)
        scores_pred = scores_pred[npoint_mask]
        proposals_pred = proposals_pred[npoint_mask]
        # semantic_id = semantic_id[npoint_mask]

        clusters = proposals_pred.numpy()
        cluster_scores = scores_pred.numpy()
        # cluster_semantic_id = semantic_id.numpy()

        nclusters = clusters.shape[0]

        pred_instances = []
        for i in range(nclusters):
            cluster_i = clusters[i]
            pred = {'scan_id': scan_id, 'label_id': semantic_pred_labels[cluster_i][0].item(), 'conf': cluster_scores[i],
                    'pred_mask': rle_encode(cluster_i)}
            pred_xyz = gt_xyz[cluster_i]
            pred['pred_bbox'] = np.concatenate((pred_xyz.min(0), pred_xyz.max(0)))
            pred_instances.append(pred)
        return pred_instances
