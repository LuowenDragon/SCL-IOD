import os
from typing import Optional, Dict

import torch
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH

from mmdet.registry import HOOKS

@HOOKS.register_module()
class SaveGTRegionFeatures(Hook):
    def after_train_epoch(self, runner) -> None:
        model = runner.model.module
        # ori_features = torch.concat(model.ori_features)
        ori_labels = torch.concat(model.ori_labels)
        feats_after_roi_align = torch.concat(model.feats_after_roi_align)
        # instance_feat_before_algin = torch.concat(model.instance_feat_before_algin)
        # instance_feat_before_after = torch.concat(model.instance_feat_before_after)
        # instance_label = torch.concat(model.instance_label)
        os.makedirs(model.save_feature_path, exist_ok=True)
        # torch.save(instance_feat_before_algin, model.save_feature_path + 'instance_feat_before_algin.pth')
        # torch.save(instance_feat_before_after, model.save_feature_path + 'instance_feat_before_after.pth')
        # torch.save(instance_label, model.save_feature_path + 'instance_label.pth')
        # torch.save(ori_features, model.save_feature_path + 'protypes_all.pth')
        torch.save(ori_labels, model.save_feature_path + 'protype_label.pth')
        torch.save(feats_after_roi_align, model.save_feature_path + 'protype.pth')