# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import re
from typing import OrderedDict

import cv2
import einops
import torch
from mmengine import Config
from mmengine.runner import load_checkpoint, load_state_dict
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
from .faster_rcnn import FasterRCNN
from ...structures import SampleList
import torch.nn.functional as F

@MODELS.register_module()
class FasterRCNNIncrease(FasterRCNN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 ori_settings: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            ori_settings=ori_settings,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        assert self.all_class_num == self.old_class_num + self.new_class_num


        self.pseudo_label_threshold = ori_settings.get('pseudo_label_threshold', 0.7)
        self.use_cross_distillation = ori_settings.get('use_cross_distillation', False)
        self.coss_distillation_weight = ori_settings.get('coss_distillation_weight', 0.2)
        self.cross_head_initialize = ori_settings.get('cross_head_initialize', False)

        self.use_protype = ori_settings.get('use_protype', False)
        self.protype_save_path = ori_settings.get('protype_save_path', None)
        self.protype_weight = ori_settings.get('protype_weight', 0.1)
        self.new_task_expert = None

        self.load_base_detector(ori_settings)
        self.frozen_others_model()

        assert self.protype_save_path is not None
        if isinstance(self.protype_save_path, list):
            protype_perclass_list = [torch.load(save_path).cuda() for save_path in self.protype_save_path]
            self.protype_perclass = torch.concat(protype_perclass_list)
        else:
            self.protype_perclass = torch.load(self.protype_save_path).cuda()
        protype_labels = [torch.full((10,), i, dtype=torch.int64) for i in range(self.old_class_num)]
        self.protype_labels = torch.concat(protype_labels).cuda()
        old_sem = self.sc[:self.old_class_num]  # [4,S]
        new_sem = self.sc[self.old_class_num:]  # [3,S]
        old_sem = F.normalize(old_sem, dim=1)
        new_sem = F.normalize(new_sem, dim=1)
        self.sim = torch.matmul(new_sem, old_sem.T)  # [3,4]
        torch.save(self.state_dict(), ori_settings['load_from_weight'])

    def frozen_others_model(self):
        for param in self.ori_model.parameters():
            param.requires_grad = False
        # if self.use_cross_distillation:
        for param in self.new_task_expert.parameters():
            param.requires_grad = False

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:

        batch_data_samples = self.filter_old_classes(batch_data_samples)

        x = self.extract_feat(batch_inputs)
        batch_data_samples_without_new = copy.deepcopy(batch_data_samples)

        with torch.no_grad():
            result_ori, rpn_results_list_old_expert, x_old_expert = self.ori_model.predict_with_roi_box(batch_inputs, copy.deepcopy(batch_data_samples), rescale=False)

        for img_index, (result, batch_data_sample) in enumerate(zip(result_ori, batch_data_samples)):
            indices = cv2.dnn.NMSBoxes(
                result.pred_instances['bboxes'].cpu().numpy(),
                result.pred_instances['scores'].cpu().numpy(),
                0.,
                0.7)
            for bbox_index, bbox in enumerate(result.pred_instances):
                if bbox['scores'] < self.pseudo_label_threshold or bbox_index not in indices:
                    continue
                # max_iou_ = self.calculate_iou_tensor(batch_data_sample.gt_instances['bboxes'], bbox['bboxes'][0])
                # if max_iou_ > 0.7:
                #     continue
                else:
                    bbox.__delattr__('scores')
                    batch_data_sample.gt_instances = batch_data_sample.gt_instances.cat(
                        [batch_data_sample.gt_instances, bbox])

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses, feature_dict = self.roi_head.loss_with_feature(x, rpn_results_list,
                                                                   batch_data_samples)
        labels = feature_dict["labels"]
        feats_before_head = feature_dict["feats_before_head"]
        feats_after_roi_align = feature_dict["feats_after_roi_align"]
        pos_inds = (labels >= 0) & (labels < self.all_class_num)

        if self.use_sc:
            x_cls = self.roi_head.bbox_head.decompose_net(feats_before_head)
            if not pos_inds.any():
                roi_losses['sc_align_loss'] = self.sc_align_loss(x_cls[:1], labels[:1] - 1) * 0.
            else:
                roi_losses['sc_align_loss'] = self.sc_align_loss(x_cls[pos_inds],
                                                                   labels[pos_inds]) * self.sc_aligned_weight
        if self.use_cross_distillation:
            result_new_task_expert, rpn_results_list_new_expert, x_new_expert = self.new_task_expert.predict_with_roi_box(batch_inputs, copy.deepcopy(batch_data_samples), rescale=False)

            latent_old_roi, scores_old = self.filter_or_match_region_to_roi(result_ori, is_match=False,
                                                                batch_data_samples_without_new=batch_data_samples_without_new)
            latent_new_roi, scores_new = self.filter_or_match_region_to_roi(result_new_task_expert, is_match=True,
                                                                batch_data_samples_without_new=batch_data_samples_without_new)

            cross_distillation_between_new_and_current = None
            cross_distillation_between_old_and_current = None

            if latent_old_roi is not None:
                old_expert_roi_features = self.ori_model.roi_head.get_roi_features(x_old_expert, latent_old_roi)
                current_model_old_roi_features = self.roi_head.get_roi_features(x, latent_old_roi)
                current_model_old_roi_features_to_current_model_logits = self.roi_head.bbox_head.decompose_net(
                    current_model_old_roi_features) @ self.sc.T
                old_expert_roi_features_to_old_model_logits = self.ori_model.roi_head.bbox_head.decompose_net(
                old_expert_roi_features) @ self.sc.T
                cross_distillation_between_old_and_current = self.logits_distillation(
                current_model_old_roi_features_to_current_model_logits,
                old_expert_roi_features_to_old_model_logits.detach(), weight=scores_old)

            if latent_new_roi is not None:
                new_expert_roi_features = self.new_task_expert.roi_head.get_roi_features(x_new_expert, latent_new_roi)
                current_model_new_roi_features = self.roi_head.get_roi_features(x, latent_new_roi)
                current_model_new_roi_features_to_current_model_logits = self.roi_head.bbox_head.decompose_net(current_model_new_roi_features) @ self.sc.T
                new_expert_roi_features_to_new_model_logits = self.new_task_expert.roi_head.bbox_head.decompose_net(new_expert_roi_features) @ self.sc.T
                cross_distillation_between_new_and_current = self.logits_distillation(
                    current_model_new_roi_features_to_current_model_logits,
                    new_expert_roi_features_to_new_model_logits.detach(), weight=scores_new)

            if cross_distillation_between_new_and_current is not None and cross_distillation_between_old_and_current is not None:
                roi_losses['cross_head_distillation_loss'] = (cross_distillation_between_new_and_current
                                                                     + cross_distillation_between_old_and_current) * self.coss_distillation_weight
            elif cross_distillation_between_new_and_current is not None:
                roi_losses['cross_head_distillation_loss'] = cross_distillation_between_new_and_current * self.coss_distillation_weight
            elif cross_distillation_between_old_and_current is not None:
                roi_losses[
                    'cross_head_distillation_loss'] = cross_distillation_between_old_and_current * self.coss_distillation_weight

        if self.use_protype:
            pseudo_protype, pseudo_labels, pseudo_prototypes_ori = self.construct_pseudo_prototypes(feats_after_roi_align[pos_inds].detach(), labels[pos_inds].detach())
            rehersal_protypes = einops.rearrange(self.protype_perclass, 'n m d h w -> (n m) d h w').detach()
            indices = torch.randperm(rehersal_protypes.shape[0])[: rehersal_protypes.shape[0] // 2]
            rehersal_protypes = rehersal_protypes[indices]
            rehersal_labels = self.protype_labels[indices]
            if pseudo_protype is not None:
                max_pseudo_num = rehersal_protypes.shape[0] // 2
                pseudo_num = min(pseudo_protype.shape[0], max_pseudo_num)
                if pseudo_num > 0:
                    indices = torch.randperm(pseudo_protype.shape[0], device=pseudo_protype.device)[:pseudo_num]
                    rehersal_protypes = torch.concat([rehersal_protypes, pseudo_protype[indices]])
                    rehersal_labels = torch.concat([rehersal_labels, pseudo_labels[indices]])
            reheasal_logits = self.roi_head.forward_roi_feature(rehersal_protypes)
            roi_losses['rehersal_loss'] = self.xent_rehersal(reheasal_logits, rehersal_labels) * self.protype_weight
        losses.update(roi_losses)

        return losses

    def filter_old_classes(self, batch_data_samples):
        for data_sample in batch_data_samples:
            gt_instances = data_sample.gt_instances

            if len(gt_instances) == 0:
                continue
            labels = gt_instances.labels
            keep_mask = labels >= self.old_class_num
            gt_instances = gt_instances[keep_mask]
            data_sample.gt_instances = gt_instances

        return batch_data_samples

    def construct_pseudo_prototypes(
            self,
            features_now,  # [B,1024]
            features_new_corresponding_labels  # [B]
    ):
        nearest_new = self.sim.argmax(dim=1)  # [3]
        pseudo_prototypes = []
        pseudo_prototypes_ori = []
        pseudo_labels = []
        unique_labels_list = torch.unique(features_new_corresponding_labels).tolist()
        for feat, label in zip(features_now, features_new_corresponding_labels):

            label = label.item()

            if label >= self.old_class_num:
                continue

            for i in range(self.new_class_num):
                # print("nearest_new:", nearest_new) [2, 0, 2]
                if nearest_new[i].item() != label:
                    continue

                if i + self.old_class_num not in unique_labels_list:
                    continue

                candidate_feat_mask = features_new_corresponding_labels == i + self.old_class_num
                true_indices = torch.nonzero(candidate_feat_mask, as_tuple=True)[0]
                random_index = true_indices[torch.randint(len(true_indices), (1,))].item()
                P_pseudo = feat.unsqueeze(0)
                P_old = self.protype_perclass[label][torch.randint(10, (1,))]
                domain_sub_vector = P_pseudo - P_old

                P_new = features_now[random_index].unsqueeze(0)

                P_new_old = P_new - domain_sub_vector
                pseudo_prototypes.append(P_new_old)
                pseudo_prototypes_ori.append(P_new)
                pseudo_labels.append(self.old_class_num + i)


        # stack
        if len(pseudo_prototypes) > 0:
            pseudo_prototypes = torch.concat(pseudo_prototypes)
            pseudo_prototypes_ori = torch.concat(pseudo_prototypes_ori)
            pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.int64).cuda()
        else:
            pseudo_prototypes = None
            pseudo_labels = None
            pseudo_prototypes_ori = None
        return pseudo_prototypes, pseudo_labels, pseudo_prototypes_ori

    def filter_or_match_region_to_roi(self, predict_result, is_match, batch_data_samples_without_new, confidence_threshold=0.2):
        roi = []
        scores = []
        for img_index, (result, batch_data_sample) in enumerate(zip(predict_result, batch_data_samples_without_new)):
            indices = cv2.dnn.NMSBoxes(
                result.pred_instances['bboxes'].cpu().numpy(),
                result.pred_instances['scores'].cpu().numpy(),
                0.,
                0.7)
            for bbox_index, bbox in enumerate(result.pred_instances):
                if bbox_index not in indices:
                    continue
                if bbox['scores'] < confidence_threshold:
                    continue
                max_iou_ = self.calculate_iou_tensor(batch_data_sample.gt_instances['bboxes'], bbox['bboxes'][0])
                if is_match and max_iou_ > 0.7:
                    scores.append(bbox['scores'])
                    roi.append(torch.concat([torch.tensor([float(img_index)]).cuda(),bbox['bboxes'][0]]))
                    continue
                if not is_match and max_iou_ < 0.7:
                    scores.append(bbox['scores'])
                    roi.append(torch.concat([torch.tensor([float(img_index)]).cuda(), bbox['bboxes'][0]]))
                    continue
        if len(roi) > 0:
            roi = torch.stack(roi,dim=0)
            scores = torch.stack(scores)
        else:
            roi = None
        return roi, scores


    def logits_distillation(self, teacher_logits, student_logits, temperature=1.0, weight=None):
        """
        KL-based logits distillation.

        Args:
            teacher_logits: Tensor [B, C]
            student_logits: Tensor [B, C]
            temperature: distillation temperature

        Returns:
            loss_kd
        """

        T = temperature

        # teacher distribution
        teacher_prob = F.softmax(teacher_logits / T, dim=1)

        # student log distribution
        student_log_prob = F.log_softmax(student_logits / T, dim=1)

        loss = F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction="none"
        )

        loss = loss.sum(dim=1)
        if weight is not None:
            # print("weight: {}".format(weight))
            loss = loss * weight
        loss_kd = loss.mean() * (T * T)
        return loss_kd

    def logits_distillation_without_softmax(self, teacher_logits, student_logits):
        loss = -(teacher_logits * student_logits.log()).sum(dim=1).mean()
        return loss


    def extract_high_score_rois(self, rpn_results_list, score_thr=0.7):
        """
        Args:
            rpn_results_list: list[InstanceData]
            score_thr: score threshold

        Returns:
            roi_tensor: Tensor [N,5]
        """

        roi_list = []

        for img_id, result in enumerate(rpn_results_list):

            scores = result.scores
            bboxes = result.bboxes

            mask = scores > score_thr

            if mask.sum() == 0:
                continue

            selected_boxes = bboxes[mask]   # [K,4]

            img_ids = torch.full(
                (selected_boxes.shape[0], 1),
                img_id,
                dtype=selected_boxes.dtype,
                device=selected_boxes.device
            )

            rois = torch.cat([img_ids, selected_boxes], dim=1)  # [K,5]

            roi_list.append(rois)

        if len(roi_list) == 0:
            return torch.empty((0,5), device=rpn_results_list[0].bboxes.device)

        roi_tensor = torch.cat(roi_list, dim=0)

        return roi_tensor

    def calculate_iou_tensor(self, gt_boxes, pred_box):
        """
        gt_boxes: Tensor of shape [M, 4] on CUDA
        pred_box: Tensor of shape [4] on CUDA
        return: Tensor scalar (max IoU)
        """
        if gt_boxes.shape[0] < 1:
            return 0.
        # Expand pred_box to [M, 4]
        pred_box = pred_box.unsqueeze(0).expand_as(gt_boxes)  # [M, 4]

        # Intersection
        x1 = torch.max(gt_boxes[:, 0], pred_box[:, 0])
        y1 = torch.max(gt_boxes[:, 1], pred_box[:, 1])
        x2 = torch.min(gt_boxes[:, 2], pred_box[:, 2])
        y2 = torch.min(gt_boxes[:, 3], pred_box[:, 3])

        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        intersection = inter_w * inter_h

        area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        area_pred = (pred_box[:, 2] - pred_box[:, 0]) * (pred_box[:, 3] - pred_box[:, 1])
        union = area_gt + area_pred - intersection
        ious = intersection / (union + 1e-6)  # [M]
        return ious.max()

    def load_base_detector(self, ori_setting):
        self._load_checkpoint_for_current_model(ori_setting.ori_checkpoint_file, ori_setting=ori_setting)

        # if self.use_cross_distillation:
        new_task_expert_cfg = Config.fromfile(ori_setting['new_task_expert_config'])
        if hasattr(new_task_expert_cfg.model, 'latest_model_flag'):
            new_task_expert_cfg.model.latest_model_flag = False
        new_task_expert = MODELS.build(new_task_expert_cfg.model)
        print('loading new expert from {}'.format(ori_setting.new_task_export_weight))
        load_checkpoint(new_task_expert, ori_setting.new_task_export_weight, strict=True)

        new_task_expert.eval()
        new_task_expert.ori_model = None
        self.new_task_expert = new_task_expert

        ori_cfg = Config.fromfile(ori_setting['ori_config_file'])
        if hasattr(ori_cfg.model, 'latest_model_flag'):
            ori_cfg.model.latest_model_flag = False
        ori_model = MODELS.build(ori_cfg.model)
        # load checkpoint
        print('loading base detector from {}'.format(ori_setting.ori_checkpoint_file))
        load_checkpoint(ori_model, ori_setting.ori_checkpoint_file, strict=True)

        ori_model.eval()
        ori_model.ori_model = None
        self.ori_model = ori_model


    def _load_checkpoint_for_current_model(self, checkpoint_file, map_location=None, ori_setting=None):
        # load ckpt
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        # get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(checkpoint_file))
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k,
                                       v in checkpoint['state_dict'].items()}


        added_branch_weight = torch.mean(state_dict['roi_head.bbox_head.fc_cls.weight'][:self.old_class_num], dim=0,
                       keepdim=True).expand(self.all_class_num - self.old_class_num, -1)
        added_branch_bias = torch.mean(state_dict['roi_head.bbox_head.fc_cls.bias'][:self.old_class_num], dim=0,
                       keepdim=True).expand(self.all_class_num - self.old_class_num)
        # modify cls head size of state_dict

        state_dict['roi_head.bbox_head.fc_cls.weight'] = torch.cat(
            (
                state_dict['roi_head.bbox_head.fc_cls.weight'][:self.old_class_num],
                added_branch_weight,
                state_dict['roi_head.bbox_head.fc_cls.weight'][-1:]
            ), dim=0)

        state_dict['roi_head.bbox_head.fc_cls.bias'] = torch.cat(
            (
                state_dict['roi_head.bbox_head.fc_cls.bias'][:self.old_class_num],
                added_branch_bias,
                state_dict['roi_head.bbox_head.fc_cls.bias'][-1:]
            ), dim=0)


        added_branch_weight = state_dict['roi_head.bbox_head.fc_reg.weight']
        added_branch_bias = state_dict['roi_head.bbox_head.fc_reg.bias']
        added_branch_weight = torch.mean(einops.rearrange(added_branch_weight, '(n m) d-> n m d', m=4), dim=0, keepdim=True)
        added_branch_bias = torch.mean(einops.rearrange(added_branch_bias, '(n m)-> n m', m=4),dim=0, keepdim=True)

        added_branch_weight = added_branch_weight.expand(self.all_class_num - self.old_class_num, -1, -1)
        added_branch_bias = added_branch_bias.expand(self.all_class_num - self.old_class_num, -1)

        added_branch_weight = einops.rearrange(added_branch_weight, 'n m d -> (n m) d')
        added_branch_bias = einops.rearrange(added_branch_bias, 'n m-> (n m)')

        state_dict['roi_head.bbox_head.fc_reg.weight'] = torch.cat(
            (state_dict['roi_head.bbox_head.fc_reg.weight'][:self.old_class_num * 4], added_branch_weight), dim=0)
        state_dict['roi_head.bbox_head.fc_reg.bias'] = torch.cat(
            (state_dict['roi_head.bbox_head.fc_reg.bias'][:self.old_class_num * 4], added_branch_bias), dim=0)

        state_dict_new = {}
        for k, v in state_dict.items():
            if k.startswith('ori_model'):
                continue
            if k.startswith('new_task_expert'):
                continue
            state_dict_new[k] = v

        # load state_dict
        if hasattr(self, 'module'):
            load_state_dict(self.module, state_dict_new)
        else:
            load_state_dict(self, state_dict_new)
