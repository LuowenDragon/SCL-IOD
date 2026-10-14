_base_ = [
    '../nighttimeSunny1_4.py'
]

base_dir='work_dir/faster_rcnn_increase/NS-DS/SCL-IOD'
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
load_from = base_dir + '/nighttimeSunny1_4/epoch_12.pth'
custom_hooks = [
    dict(type='SaveGTRegionFeatures'),]

model = dict(
    type='FasterRCNNGetProtype',
    ori_settings=dict(
        use_sc=True,
        sc_aligned_weight=0.1,
        sc_weight_path='./semantic_matrix/BDD7_clip_rn50.pth',
        save_feature_path=base_dir + '/nighttimeSunny1_4/prototype/'
    ),)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=10,
        by_epoch=True,
        milestones=[8, 10],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.000001, momentum=0.9, weight_decay=0.0001))

