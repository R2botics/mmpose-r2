"""DEPTH ABLATION arm: 5 channels -- RGB + height + validity mask.

One of three configs that differ ONLY in the input channel count. Everything
else -- backbone, decoder, codec, sigma, loss, schedule, splits, evaluator,
selection rule, augmentation and its RNG draw order -- is inherited verbatim
from `vitpose-b_8-kp_udp_256x256.py`. `scripts/verify_vitpose_ablation.py`
asserts that mechanically, so "only depth changed" is a checked claim rather
than an intention.

Run all three from the SAME converted pretrain. Because extra channels go
through `ViT.extra_stem` (a separate zero-initialised projection) rather than
a widened patch embed, the RGB tokenizer is bit-identical across the three arms
at step 0 -- the 5ch model literally starts as the 3ch model.

Compare on `max/cyclic_corner_span_err_px` and `frac_collapsed_pairs`, per
domain and never pooled. The brief's own history is the reason: RGB-only won
on the real val set and lost 15 PCK points out of distribution, so the ranking
inverts depending on which domain you read. The OOD domain is reported here
with `select=False` -- it is the tiebreak you look at, not the one you select
on.
"""
_base_ = ['./vitpose-b_8-kp_udp_256x256.py']

in_channels = 5
img_mean = [123.675, 116.28, 103.53, 414.2, 127.5]
img_std = [58.395, 57.12, 57.375, 184.9, 127.5]

model = dict(
    data_preprocessor=dict(mean=img_mean, std=img_std),
    backbone=dict(in_channels=in_channels))

# The pipelines must be restated in full: mmengine replaces a list wholesale
# rather than merging it, so overriding the base's `train_pipeline` variable
# would not reach the dataloader that already embedded it. Keep these in sync
# with the base by hand -- `scripts/verify_vitpose_ablation.py` fails the build
# if they drift in anything but the channel count.
train_pipeline = [
    dict(type='LoadRGBDImage', channels=in_channels),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomBBoxTransform',
         shift_factor=0.1, scale_factor=[0.75, 1.25], rotate_factor=30),
    dict(type='TopdownAffine', input_size=(256, 256), use_udp=True),
    dict(type='PhotometricDistortionRGBOnly',
         brightness_delta=32, contrast_range=(0.6, 1.4),
         saturation_range=(0.6, 1.4), hue_delta=18),
    dict(type='GenerateTarget', encoder={{_base_.codec}}),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadRGBDImage', channels=in_channels),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256), use_udp=True),
    dict(type='PackPoseInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=val_pipeline))
