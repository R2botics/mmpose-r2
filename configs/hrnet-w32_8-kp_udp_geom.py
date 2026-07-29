"""RGB-only counterpart to hrnet-w32_8-kp_udp_rgbd_geom.py.

This is the control arm for answering "does the depth channel earn its place?".
Everything is held identical to the RGBD+geom config except the input
modality — same codec, same geometric loss and weights, same schedule — so the
existing RGBD runs are a valid comparison.

Differences from the RGBD lineage, all forced by dropping to 3 channels:
  * LoadImage instead of LoadRGBDImage
  * stock PhotometricDistortion instead of PhotometricDistortionRGBOnly
    (the RGB-only variant exists only because the standard one cannot handle a
    5-channel array)
  * stock PoseDataPreprocessor, not NChannelPoseDataPreprocessor
  * backbone in_channels=3, initialised from stock ImageNet HRNet-W32 — no
    5-channel inflation step (scripts/convert_hrnet_to_rgbd.py) is needed

Data note: this points at data/RSC_Keypoints_RGBD, NOT data/RSC_Keypoints. That
dataset already holds the colour images under images/ and every annotation
file under annotations/ — including the synthetic split — so the RGB arm reuses
it directly and just never reads depth/. Keeping both arms on one data root
also guarantees the train/val splits are identical.
"""
_base_ = ['./hrnet-w32_8-kp_udp.py']

custom_imports = dict(
    imports=[
        # Only the loss is needed here — the rgbd transforms and the N-channel
        # preprocessor are deliberately not imported.
        'mmpose.models.losses.geometric_loss',
    ],
    allow_failed_imports=False)

data_root = 'data/RSC_Keypoints_RGBD'
metainfo = 'configs/_base_/datasets/RSC_Keypoints.py'

# Only change vs the RGB UDP config: swap the head's loss for the geometric
# one. Weights are copied verbatim from hrnet-w32_8-kp_udp_rgbd_geom.py so the
# two arms differ in modality alone.
model = dict(
    head=dict(
        loss=dict(
            _delete_=True,
            type='GeometricKeypointMSELoss',
            use_target_weight=True,
            perp_weight=0.005,
            parallel_weight=0.005,
            tolerance_deg=10.0,
            softmax_temp=10.0,
            warmup_steps=1800)))

# The base config baked 'data/RSC_Keypoints' into these dicts, so assigning
# `data_root` above is not enough — the nested keys must be overridden too.
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/person_keypoints_train.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/person_keypoints_val.json'))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}/annotations/person_keypoints_val.json')
