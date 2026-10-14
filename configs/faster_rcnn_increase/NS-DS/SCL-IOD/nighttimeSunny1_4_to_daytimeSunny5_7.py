_base_ = [
    '../nighttimeSunny1_4_to_daytimeSunny5_7.py'
]

base_dir='work_dir/faster_rcnn_increase/NS-DS/SCL-IOD/'
work_dir= base_dir + 'nighttimeSunny1_4_to_daytimeSunny5_7'

ori_checkpoint = base_dir + 'nighttimeSunny1_4/epoch_12.pth'
ori_config_file = base_dir + 'nighttimeSunny1_4/nighttimeSunny1_4.py'

new_task_expert_config = base_dir + '/daytimeSunny5_7/daytimeSunny5_7.py'
new_task_export_weight = base_dir + '/daytimeSunny5_7/epoch_12.pth'

load_from_weight = './temp_cheakpoints/temp.pth'
load_from = load_from_weight

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=7, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


model = dict(
    type='FasterRCNNIncrease',
    ori_settings=dict(
        load_from_weight=load_from_weight,
        # Baseline
        pseudo_label_threshold=0.4,
        # ETF
        use_sc=True,
        sc_weight_path='./semantic_matrix/BDD7_clip_rn50.pth',
        sc_aligned_weight=0.1,
        ori_checkpoint_file=ori_checkpoint,
        ori_config_file=ori_config_file,
        # Cross Head Distillation
        use_cross_distillation=True,
        coss_distillation_weight=0.1,
        new_task_expert_config=new_task_expert_config,
        new_task_export_weight=new_task_export_weight,
        cross_head_initialize=False,
        # Protype
        use_protype=True,
        protype_save_path=base_dir + 'nighttimeSunny1_4/prototype/class_centroids_10_per_class.pt',
        protype_weight=0.5,
    ),)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=10,
        by_epoch=True,
        milestones=[5],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

