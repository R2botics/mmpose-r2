_base_ = ['./hrnet-w32_8-kp.py']

# Register custom transforms + the N-channel data preprocessor
custom_imports = dict(
    imports=[
        'mmpose.datasets.transforms.rgbd',
        'mmpose.models.data_preprocessors.nchannel',
    ],
    allow_failed_imports=False)

# Codec — same UDP setup that gave us 0.9824 with RGB-only
codec = dict(
    type='UDPHeatmap',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2)

# Model — backbone now takes 5-channel input, 5-channel normalization
model = dict(
    data_preprocessor=dict(
        _delete_=True,
        type='NChannelPoseDataPreprocessor',
        # Channels:           R       G       B      depth   mask
        mean=[123.675, 116.28, 103.53, 414.2, 127.5],
        std=[58.395, 57.12, 57.375, 184.9, 127.5]),
    backbone=dict(
        in_channels=5,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='work_dirs/pretrained/hrnet_w32_rgbd.pth')),
    head=dict(decoder=codec))

# Dataset paths
data_mode = 'topdown'
dataset_type = 'CocoDataset'
data_root = 'data/RSC_Keypoints_RGBD'
metainfo = 'configs/_base_/datasets/RSC_Keypoints.py'

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
            dict(type='LoadRGBDImage'),           # custom: 5-channel RGBD+mask
            dict(type='GetBBoxCenterScale'),
            dict(
                type='RandomBBoxTransform',
                shift_factor=0.1,
                scale_factor=[0.75, 1.25],
                rotate_factor=30),
            dict(type='TopdownAffine', input_size=(256, 256), use_udp=True),
            dict(
                type='PhotometricDistortionRGBOnly',  # custom: RGB-only jitter
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
            dict(type='LoadRGBDImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(256, 256), use_udp=True),
            dict(type='PackPoseInputs')
        ]))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}/annotations/person_keypoints_val.json')
