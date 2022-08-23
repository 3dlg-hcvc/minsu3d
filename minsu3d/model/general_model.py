from minsu3d.evaluation.instance_segmentation import GeneralDatasetEvaluator
from minsu3d.evaluation.object_detection import evaluate_bbox_acc
from minsu3d.optimizer import init_optimizer, cosine_lr_decay
from minsu3d.common_ops.functions import common_ops
from minsu3d.loss import PTOffsetLoss, SemSegLoss
from minsu3d.model.module import Backbone
from minsu3d.util import save_prediction
import pytorch_lightning as pl
import numpy as np
import torch
import os


class GeneralModel(pl.LightningModule):
    def __init__(self, model, data, optimizer, lr_decay, inference=None):
        super().__init__()
        self.save_hyperparameters()
        input_channel = model.use_coord * 3 + model.use_color * 3 + model.use_normal * 3 + model.use_multiview * 128
        self.backbone = Backbone(input_channel=input_channel,
                                 output_channel=model.m,
                                 block_channels=model.blocks,
                                 block_reps=model.block_reps,
                                 sem_classes=data.classes)

    def configure_optimizers(self):
        return init_optimizer(parameters=self.parameters(), **self.hparams.optimizer)

    def _feed(self, data_dict):
        if self.hparams.model.use_coord:
            data_dict["feats"] = torch.cat((data_dict["feats"], data_dict["locs"]), dim=1)
        data_dict["voxel_feats"] = common_ops.voxelization(data_dict["feats"], data_dict["p2v_map"]) # (M, C), float, cuda
        output_dict = self.forward(data_dict)
        return output_dict

    def forward(self, data_dict):
        backbone_output_dict = self.backbone(data_dict["voxel_feats"], data_dict["voxel_locs"], data_dict["v2p_map"])
        return backbone_output_dict

    def _loss(self, data_dict, output_dict):
        losses = {}
        total_loss = 0

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
        offset_norm_loss, offset_dir_loss = pt_offset_criterion(output_dict["point_offsets"], gt_offsets,
                                                                valid_mask=valid)
        losses["offset_norm_loss"] = offset_norm_loss
        losses["offset_dir_loss"] = offset_dir_loss
        return losses, total_loss

    def training_step(self, data_dict, idx):
        # prepare input and forward
        output_dict = self._feed(data_dict)
        losses, total_loss = self._loss(data_dict, output_dict)
        self.log("train/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.hparams.data.batch_size)
        for key, value in losses.items():
            self.log(f"train/{key}", value, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.hparams.data.batch_size)
        return total_loss

    def training_epoch_end(self, training_step_outputs):
        cosine_lr_decay(self.trainer.optimizers[0], self.hparams.optimizer.lr, self.current_epoch,
                        self.hparams.lr_decay.decay_start_epoch, self.hparams.lr_decay.decay_stop_epoch, 1e-6)

    def validation_step(self, data_dict, idx):
        pass

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
        pass

    def test_epoch_end(self, results):
        # evaluate instance predictions
        if self.current_epoch > self.hparams.model.prepare_epochs:
            all_pred_insts = []
            all_gt_insts = []
            all_gt_insts_bbox = []
            all_sem_acc = []
            all_sem_miou = []
            inference_time = 0
            for semantic_accuracy, semantic_mean_iou, pred_instances, gt_instances, gt_instances_bbox, end_time in results:
                all_sem_acc.append(semantic_accuracy)
                all_sem_miou.append(semantic_mean_iou)
                all_gt_insts_bbox.append(gt_instances_bbox)
                all_pred_insts.append(pred_instances)
                all_gt_insts.append(gt_instances)
                inference_time += end_time
            self.print(f"Average inference time: {round(inference_time / len(results), 3)}s per scan.")
            if self.hparams.inference.save_predictions:
                save_prediction(self.hparams.inference.output_dir, all_pred_insts, self.hparams.data.mapping_classes_ids)
                self.print(f"\nPredictions saved at {os.path.join(self.hparams.inference.output_dir, 'instance')}\n")

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

