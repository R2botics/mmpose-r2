_base_ = ['mmpose::_base_/default_runtime.py']

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='coco/AP',
        rule='greater'))

# Override visualizer to add TensorBoard logging
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# Codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# Model Configuration
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        deepen_factor=0.33,
        widen_factor=0.5,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth'
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=512,
        out_channels=8,
        input_size=(256, 256),
        in_featuremap_size=(8, 8),
        loss=dict(type='KLDiscretLoss', use_target_weight=True, label_softmax=True),
        decoder=codec)
)

# Optimizer Wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        ))

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
metainfo = 'configs/_base_/datasets/RSC_Keypoints.py'

train_dataloader = dict(
    batch_size=8,
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
    batch_size=8,
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
