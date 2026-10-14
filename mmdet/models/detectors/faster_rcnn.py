# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
from ...structures import SampleList
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

@MODELS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 ori_settings: ConfigType = None,
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
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.old_class_num = ori_settings['old_class_num']
        self.new_class_num = ori_settings['new_class_num']
        self.all_class_num = ori_settings['num_class']
        self.use_sc = ori_settings.get('use_sc', False)
        self.sc = None
        self.xent_sc_align = None
        self.sc_weight_path = ori_settings.get('sc_weight_path', None)
        self.sc_aligned_weight = ori_settings.get('sc_aligned_weight', 0.)
        self.old_class_dont_count = ori_settings.get('old_class_dont_count', 0)
        self.xent_rehersal = CrossEntropyLabelSmooth(num_classes=self.all_class_num + 1)
        assert self.all_class_num == self.old_class_num + self.new_class_num

        assert self.sc_weight_path is not None
        self.initial_sc()
        if not self.use_sc:
            for param in self.roi_head.bbox_head.decompose_net.parameters():
                param.requires_grad = False
        self.cheack_head_dim()

    def cheack_head_dim(self):
        print("head dim: {}".format(self.roi_head.bbox_head.fc_cls.weight.shape[0]))
        assert self.roi_head.bbox_head.fc_cls.weight.shape[0] == self.all_class_num + 1

    def re_label_when_is_temp_model(self, batch_data_samples):
        for data_sample in batch_data_samples:
            data_sample.gt_instances.labels = data_sample.gt_instances.labels - self.old_class_num

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)

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
        pos_inds = (labels >= 0) & (labels < self.all_class_num)
        # all_are_4 = torch.all(labels[pos_inds] == 4).item()
        # print("labels: ", labels[pos_inds])
        # if not all_are_4:
        #     assert 0

        if self.use_sc:
            x_cls = self.roi_head.bbox_head.decompose_net(feats_before_head)
            if not pos_inds.any():
                roi_losses['sc_align_loss'] = self.sc_align_loss(x_cls[:1], labels[:1] - 1) * 0.
            else:
                roi_losses['sc_align_loss'] = self.sc_align_loss(x_cls[pos_inds], labels[pos_inds]) * self.sc_aligned_weight
        losses.update(roi_losses)

        return losses

    def sc_align_loss(self, feat_pos: Tensor, label_pos: Tensor):
        # print(self.sc[:, 0])
        # print(label_pos + self.old_class_dont_count)
        logit = feat_pos @ self.sc.T
        loss = self.xent_sc_align(logit, label_pos + self.old_class_dont_count)
        return loss

    def initial_sc(self):
        self.sc = torch.load(self.sc_weight_path).cuda()[:self.old_class_dont_count + self.all_class_num]
        print("sc dim: {}, sc begin: {}, sc end: {}".format(self.sc.shape, 0, self.old_class_dont_count + self.all_class_num))
        self.xent_sc_align = CrossEntropyLabelSmooth(num_classes=self.sc.shape[0])


    def predict_with_roi_box(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples, rpn_results_list, x