_base_ = ['mmpose::_base_/default_runtime.py']

# Override visualizer to add TensorBoard logging
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='coco/AP',
        rule='greater'))

# Codec settings — MSRAHeatmap outputs 64x64 heatmaps for 256x256 input
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2)

# Model Configuration
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth')),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=8,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=False))

# Optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4))

# Training Schedule
train_cfg = dict(by_epoch=True, max_epochs=700, val_interval=10)
val_cfg = dict()
test_cfg = None

# Learning Rate Scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=100),
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        begin=0,
        end=700,
        by_epoch=True),
]

# Dataset & Dataloader
data_mode = 'topdown'
dataset_type = 'CocoDataset'
data_root = 'data/RSC_Keypoints'
metainfo = '/opt/nuclio/RSC_Keypoints.py'

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/person_keypoints_train.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(from_file=metainfo),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(
                type='RandomBBoxTransform',
                shift_factor=0.1,
                scale_factor=[0.75, 1.25],
                rotate_factor=30),
            dict(type='TopdownAffine', input_size=(256, 256), use_udp=True),
            dict(
                type='PhotometricDistortion',
                brightness_delta=32,
                contrast_range=(0.6, 1.4),
                saturation_range=(0.6, 1.4),
                hue_delta=18),
            dict(type='GenerateTarget', encoder=codec),
            dict(type='PackPoseInputs')
        ]))

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/person_keypoints_val.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(from_file=metainfo),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(256, 256), use_udp=True),
            dict(type='PackPoseInputs')
        ]))

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/RSC_Keypoints/annotations/person_keypoints_val.json')

test_dataloader = None
