import torch
import torch.nn as nn
import pytorch_lightning as pl
from lib.evaluation.instance_seg_helper import ScanNetEval, rle_encode, get_gt_instances
from lib.common_ops.functions import softgroup_ops
from lib.common_ops.functions import common_ops
from lib.loss import *
from lib.evaluation.semantic_seg_helper import *
from model.module import Backbone, TinyUnet
from model.helper import clusters_voxelization, get_batch_offsets
from lib.optimizer import init_optimizer


class SoftGroup(pl.LightningModule):
    def __init__(self, **cfg):
        super().__init__()
        self.save_hyperparameters()
        input_channel = cfg["model"].use_coords * 3 + cfg["model"].use_color * 3 + cfg["model"].use_normal * 3
        output_channel = cfg["model"].m
        semantic_classes = cfg["data"].classes
        self.instance_classes = semantic_classes - len(cfg["data"].ignore_classes)

        """
            Backbone Block
        """
        self.backbone = Backbone(input_channel=input_channel,
                                 output_channel=cfg["model"].m,
                                 block_channels=cfg["model"].blocks,
                                 block_reps=cfg["model"].block_reps,
                                 sem_classes=semantic_classes)

        """
            Top-down Refinement Block
        """
        self.tiny_unet = TinyUnet(output_channel)

        self.classification_branch = nn.Linear(output_channel, self.instance_classes + 1)

        self.mask_scoring_branch = nn.Sequential(
            nn.Linear(output_channel, output_channel),
            nn.ReLU(inplace=True),
            nn.Linear(output_channel, self.instance_classes + 1)
        )

        self.iou_score = nn.Linear(output_channel, self.instance_classes + 1)

    def forward(self, data_dict):
        output_dict = {}
        """
            Bottom-up Grouping Block
        """
        backbone_output_dict = self.backbone(data_dict["voxel_feats"], data_dict["voxel_locs"], data_dict["v2p_map"])
        output_dict.update(backbone_output_dict)

        if self.current_epoch > self.hparams.model.prepare_epochs or self.hparams.model.freeze_backbone:
            """
                Top-down Refinement Block
            """
            semantic_scores = output_dict["semantic_scores"].softmax(dim=-1)
            batch_idxs = data_dict["vert_batch_ids"].int()


            proposals_offset_list = []
            proposals_idx_list = []

            for class_id in range(self.hparams.data.classes):
                if class_id in self.hparams.data.ignore_classes:
                    continue
                scores = semantic_scores[:, class_id].contiguous()
                object_idxs = (scores > self.hparams.model.grouping_cfg.score_thr).nonzero().view(-1)
                if object_idxs.size(0) < self.hparams.model.test_cfg.min_npoint:
                    continue
                batch_idxs_ = batch_idxs[object_idxs]
                batch_offsets_ = get_batch_offsets(batch_idxs_, self.hparams.data.batch_size, self.device)
                coords_ = data_dict["locs"][object_idxs]
                pt_offsets_ = output_dict["point_offsets"][object_idxs]
                idx, start_len = common_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_,
                                                                 self.hparams.model.grouping_cfg.radius, self.hparams.model.grouping_cfg.mean_active)

                proposals_idx, proposals_offset = softgroup_ops.sg_bfs_cluster(self.hparams.model.grouping_cfg.class_numpoint_mean, idx.cpu(),
                                                                            start_len.cpu(),
                                                                            self.hparams.model.grouping_cfg.npoint_thr, class_id)
                proposals_idx[:, 1] = object_idxs.cpu()[proposals_idx[:, 1].long()].int()

                # merge proposals
                if len(proposals_offset_list) > 0:
                    proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                    proposals_offset += proposals_offset_list[-1][-1]
                    proposals_offset = proposals_offset[1:]
                if proposals_idx.size(0) > 0:
                    proposals_idx_list.append(proposals_idx)
                    proposals_offset_list.append(proposals_offset)
            proposals_idx = torch.cat(proposals_idx_list, dim=0)
            proposals_offset = torch.cat(proposals_offset_list)

            if proposals_offset.shape[0] > self.hparams.model.train_cfg.max_proposal_num:
                proposals_offset = proposals_offset[:self.hparams.model.train_cfg.max_proposal_num + 1]
                proposals_idx = proposals_idx[:proposals_offset[-1]]
                assert proposals_idx.shape[0] == proposals_offset[-1]

            proposals_offset = proposals_offset.cuda()
            output_dict["proposals_idx"] = proposals_idx
            output_dict["proposals_offset"] = proposals_offset

            inst_feats, inst_map = clusters_voxelization(
                clusters_idx=proposals_idx,
                clusters_offset=proposals_offset,
                feats=output_dict["point_features"],
                coords=data_dict["locs"],
                mode=4,
                device=self.device,
                **self.hparams.model.instance_voxel_cfg
            )

            inst_map = inst_map.long().cuda()

            feats = self.tiny_unet(inst_feats)

            # predict mask scores
            mask_scores = self.mask_scoring_branch(feats.features)
            output_dict["mask_scores"] = mask_scores[inst_map]
            output_dict["instance_batch_idxs"] = feats.coordinates[:, 0][inst_map]

            # predict instance cls and iou scores
            feats = self.global_pool(feats)
            output_dict["cls_scores"] = self.classification_branch(feats)
            output_dict["iou_scores"] = self.iou_score(feats)

        return output_dict

    def global_pool(self, x, expand=False):
        indices = x.coordinates[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1,), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = softgroup_ops.global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool
        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x

    def configure_optimizers(self):
        return init_optimizer(parameters=self.parameters(), **self.hparams.train.optim)

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
        gt_offsets = data_dict["instance_info"][:, 0:3] - data_dict["locs"]  # (N, 3)
        valid = data_dict["instance_ids"] != self.hparams.data.ignore_label
        pt_offset_criterion = PTOffsetLoss()
        offset_norm_loss, offset_dir_loss = pt_offset_criterion(output_dict["point_offsets"], gt_offsets, valid_mask=valid)
        losses["offset_norm_loss"] = offset_norm_loss
        # losses["offset_dir_loss"] = offset_dir_loss

        total_loss = self.hparams.train.loss_weight[0] * semantic_loss + self.hparams.train.loss_weight[1] * offset_norm_loss

        if self.current_epoch > self.hparams.model.prepare_epochs:
            proposals_idx = output_dict["proposals_idx"][:, 1].cuda()
            proposals_offset = output_dict["proposals_offset"]

            # calculate iou of clustered instance
            ious_on_cluster = softgroup_ops.get_mask_iou_on_cluster(proposals_idx, proposals_offset,
                                                                    data_dict["instance_ids"],
                                                                    data_dict["instance_num_point"])

            # filter out background instances
            fg_inds = (data_dict["instance_semantic_cls"] != self.hparams.data.ignore_label)
            fg_instance_cls = data_dict["instance_semantic_cls"][fg_inds]
            fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

            # overlap > thr on fg instances are positive samples
            max_iou, gt_inds = fg_ious_on_cluster.max(1)
            pos_inds = max_iou >= self.hparams.model.train_cfg.pos_iou_thr
            pos_gt_inds = gt_inds[pos_inds]

            """classification loss"""

            # follow detection convention: 0 -> K - 1 are fg, K is bg
            labels = fg_instance_cls.new_full((fg_ious_on_cluster.size(0),), self.instance_classes)
            labels[pos_inds] = fg_instance_cls[pos_gt_inds]
            classification_criterion = ClassificationLoss()
            classification_loss = classification_criterion(output_dict["cls_scores"], labels)
            losses["classification_loss"] = classification_loss

            """mask scoring loss"""
            mask_cls_label = labels[output_dict["instance_batch_idxs"].long()]
            slice_inds = torch.arange(0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)
            mask_scores_sigmoid_slice = output_dict["mask_scores"].sigmoid()[slice_inds, mask_cls_label]

            mask_label = softgroup_ops.get_mask_label(proposals_idx, proposals_offset, data_dict["instance_ids"],
                                                      data_dict["instance_semantic_cls"],
                                                      data_dict["instance_num_point"], ious_on_cluster,
                                                      self.hparams.model.train_cfg.pos_iou_thr)

            mask_label_weight = (mask_label != -1).float()
            mask_label[mask_label == -1.] = 0.5  # any value is ok
            mask_scoring_criterion = MaskScoringLoss(weight=mask_label_weight, reduction='sum')
            mask_scoring_loss = mask_scoring_criterion(mask_scores_sigmoid_slice, mask_label)
            mask_scoring_loss /= (mask_label_weight.sum() + 1)
            losses["mask_scoring_loss"] = mask_scoring_loss
            """iou scoring loss"""
            ious = softgroup_ops.get_mask_iou_on_pred(proposals_idx, proposals_offset, data_dict["instance_ids"],
                                                      data_dict["instance_num_point"],
                                                      mask_scores_sigmoid_slice.detach())
            fg_ious = ious[:, fg_inds]
            gt_ious, _ = fg_ious.max(1)
            slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
            iou_score_weight = labels < self.instance_classes
            iou_score_slice = output_dict["iou_scores"][slice_inds, labels]
            iou_scoring_criterion = IouScoringLoss(reduction="none")
            iou_scoring_loss = iou_scoring_criterion(iou_score_slice, gt_ious)
            iou_scoring_loss = iou_scoring_loss[iou_score_weight].sum() / (iou_score_weight.count_nonzero() + 1)
            losses["iou_scoring_loss"] = iou_scoring_loss
            total_loss += + self.hparams.train.loss_weight[3] * classification_loss + self.hparams.train.loss_weight[
                4] * mask_scoring_loss + self.hparams.train.loss_weight[5] * iou_scoring_loss

        """total loss"""
        return losses, total_loss

    def _feed(self, data_dict):
        if self.hparams.model.use_coords:
            data_dict["feats"] = torch.cat((data_dict["feats"], data_dict["locs"]), dim=1)

        data_dict["voxel_feats"] = common_ops.voxelization(data_dict["feats"], data_dict["p2v_map"],
                                                              self.hparams.data.mode)  # (M, C), float, cuda
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
            for batch, output in outputs:
                pred_instances = self._get_pred_instances(output["proposals_idx"].cpu(), output["semantic_scores"].cpu(),
                                                          output["cls_scores"].cpu(), output["iou_scores"].cpu(),
                                                          output["mask_scores"].cpu())
                gt_instances = get_gt_instances(batch["sem_labels"].cpu(), batch["instance_ids"].cpu(), self.hparams.data.ignore_classes)
                all_pred_insts.append(pred_instances)
                all_gt_insts.append(gt_instances)
            evaluator = ScanNetEval(self.hparams.data.class_names)
            evaluation_result = evaluator.evaluate(all_pred_insts, all_gt_insts)
            self.log("val_accuracy/AP", evaluation_result["all_ap"], sync_dist=True)
            self.log("val_accuracy/AP_50", evaluation_result['all_ap_50%'], sync_dist=True)
            self.log("val_accuracy/AP_25", evaluation_result["all_ap_25%"], sync_dist=True)

    def test_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self._feed(data_dict)
        return data_dict, output_dict

    def predict_step(self, data_dict, batch_idx, dataloader_idx=0):
        # prepare input and forward
        output_dict = self._feed(data_dict)
        return data_dict, output_dict

    def test_epoch_end(self, results):
        # evaluate instance predictions
        if self.current_epoch > self.hparams.model.prepare_epochs:
            all_pred_insts = []
            all_gt_insts = []
            for batch, output in results:
                pred_instances = self._get_pred_instances(output["proposals_idx"].cpu(),
                                                          output["semantic_scores"].cpu(),
                                                          output["cls_scores"].cpu(), output["iou_scores"].cpu(),
                                                          output["mask_scores"].cpu())
                gt_instances = get_gt_instances(batch["sem_labels"].cpu(), batch["instance_ids"].cpu(), self.hparams.data.ignore_classes)
                all_pred_insts.append(pred_instances)
                all_gt_insts.append(gt_instances)
            evaluator = ScanNetEval(self.hparams.data.class_names)
            evaluation_result = evaluator.evaluate(all_pred_insts, all_gt_insts, print_result=True)

    def _get_pred_instances(self, proposals_idx, semantic_scores, cls_scores, iou_scores, mask_scores):
        num_instances = cls_scores.size(0)
        num_points = semantic_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        cls_pred_list, score_pred_list, mask_pred_list = [], [], []
        for i in range(self.instance_classes):
            cls_pred = cls_scores.new_full((num_instances,), i + 1, dtype=torch.long)
            cur_cls_scores = cls_scores[:, i]
            cur_iou_scores = iou_scores[:, i]
            cur_mask_scores = mask_scores[:, i]
            score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)
            mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device="cpu")
            mask_inds = cur_mask_scores > self.hparams.model.test_cfg.mask_score_thr
            cur_proposals_idx = proposals_idx[mask_inds].long()
            mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1

            # filter low score instance
            inds = cur_cls_scores > self.hparams.model.test_cfg.cls_score_thr
            cls_pred = cls_pred[inds]
            score_pred = score_pred[inds]
            mask_pred = mask_pred[inds]

            # filter too small instances
            npoint = mask_pred.sum(1)
            inds = npoint >= self.hparams.model.test_cfg.min_npoint
            cls_pred = cls_pred[inds]
            score_pred = score_pred[inds]
            mask_pred = mask_pred[inds]
            cls_pred_list.append(cls_pred)
            score_pred_list.append(score_pred)
            mask_pred_list.append(mask_pred)
        cls_pred = torch.cat(cls_pred_list).numpy()
        score_pred = torch.cat(score_pred_list).numpy()
        mask_pred = torch.cat(mask_pred_list).numpy()

        instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            instances.append(pred)
        return instances


