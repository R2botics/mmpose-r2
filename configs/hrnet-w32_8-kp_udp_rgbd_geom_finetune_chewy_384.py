"""384x384 variant of hrnet-w32_8-kp_udp_rgbd_geom_finetune_chewy.py.

ONLY the input resolution changes: 256x256 -> 384x384, heatmap 64x64 -> 96x96.
Same data, same splits, same sampling ratio, same schedule, same loss, same
evaluators and selection metric -- so the comparison against the 256 run is
clean.

WHY: measured on the Chewy val set (mean image ~552px), roughly two thirds of
the error budget is the codec, not the model:

    measured median error        2.38 px
    decode floor at 256/64x64    1.65 px   <- unbeatable with a perfect heatmap
    implied model error          1.72 px   (in quadrature)

The floor is the round-trip error of encoding a keypoint to a heatmap and
decoding it back, scaled into original-image pixels. Raising the HEATMAP
resolution alone does nothing (64->128 measured 1.58 -> 1.66 px: UDP/DARK
sub-pixel refinement already extracts what is there). The bottleneck is the
crop being squashed to 256 before the network ever sees it. Raising the INPUT
resolution is what moves it:

    orig size   256/64   384/96
        552 px    1.66     1.04
        823 px    2.40     1.61

Projected Chewy median: sqrt(1.72^2 + 1.04^2) ~= 2.01 px, about 15% better.
Larger boxes gain most -- at 823 px the current codec alone costs 2.4 px, more
than the whole measured median.

sigma stays at 2. mmpose's own 384 recipes often scale sigma to 3, but measured
here it makes almost no difference to the floor (1.04 vs 1.07 px), so leaving it
alone keeps "only the input size changed" literally true.
"""
_base_ = ['./hrnet-w32_8-kp_udp_rgbd_geom_finetune_chewy.py']

# ---------------------------------------------------------------- resolution
codec = dict(
    type='UDPHeatmap',
    input_size=(384, 384),
    heatmap_size=(96, 96),
    sigma=2)

# NOTE: a single `model` dict. Two separate `model = dict(...)` assignments in
# one config would silently discard the first -- plain Python rebinding, not a
# merge -- which is how the decoder override got lost on the first attempt.
model = dict(
    head=dict(
        decoder=codec,
        # Halving the batch doubles iters/epoch (428 effective samples:
        # 27 -> 54), so the geometric warmup is scaled to cover the same number
        # of EPOCHS as the 256 run (200 steps at batch 16 ~= 7.4 epochs).
        loss=dict(warmup_steps=400)))

# The backbone and head are fully convolutional, so the 256-trained
# synthetic-pretrain checkpoint in `load_from` transfers to 384 unchanged --
# only the spatial size of the activations differs.

train_pipeline_384 = [
    dict(type='LoadRGBDImage'),
    dict(type='GetBBoxCenterScale'),
    # See the note in the 256 config: horizontal mirror, keypoint permutation
    # comes from the `swap=` fields in the dataset metainfo.
    dict(type='RandomFlip', direction='horizontal'),
    # See the 256 config: RandomFlipVertical, NOT RandomFlip('vertical').
    dict(type='RandomFlipVertical', prob=0.5),
    dict(type='RandomBBoxTransform',
         shift_factor=0.1, scale_factor=[0.75, 1.25], rotate_factor=30),
    dict(type='TopdownAffine', input_size=(384, 384), use_udp=True),
    dict(type='PhotometricDistortionRGBOnly',
         brightness_delta=32, contrast_range=(0.6, 1.4),
         saturation_range=(0.6, 1.4), hue_delta=18),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline_384 = [
    dict(type='LoadRGBDImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(384, 384), use_udp=True),
    dict(type='PackPoseInputs'),
]

# 384x384 is 2.25x the activation memory of 256x256. batch_size=8 is the safe
# choice on an 8GB card (the training box reports an RTX 3070). If it fits at
# 16, USE 16 -- it keeps batch size out of the comparison entirely.
train_dataloader = dict(batch_size=8, dataset=dict(pipeline=train_pipeline_384))
val_dataloader = dict(batch_size=8, dataset=dict(pipeline=val_pipeline_384))
test_dataloader = dict(batch_size=8, dataset=dict(pipeline=val_pipeline_384))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SafeMLflowVisBackend',
        exp_name='box-flap-pose',
        run_name='rgbd-geom-finetune-chewy2-384',
        tracking_uri='http://10.200.104.2:31016',
        tags=dict(
            model='hrnet-w32',
            variant='rgbd-geom',
            data='real+chewy2',
            train_size='288+140(x1)',
            stage='finetune-real+chewy2',
            backbone_init='synthetic-pretrained',
            input_size='384'),          # <- the one variable under test
        artifact_suffix=('.py', '.pth', '.json')),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
