"""SIGMA EXPERIMENT: identical to the chewy finetune except sigma 2.0 -> 1.5.

WHY
    The two keypoints marking one physical box corner (e.g. EAST_2 and
    SOUTH_1) are routinely collapsed onto a single point. Measured on
    Chewy + OOD with best_coco_AP_epoch_120:

        6% of corner-pairs collapse (predicted separation < half of true)
        but they carry 16% of ALL keypoint error, because the collapse rate
        rises with true separation:  4% under 10px  ->  22% over 40px.

    Worst case: 48123 EAST_2, whose prediction landed 4.9px from SOUTH_1's
    ground truth and 65.1px from its own, on a pair genuinely 61.8px apart.

    The cause is resolution relative to sigma. Corner-pair separation measured
    in HEATMAP pixels across Chewy + OOD + RSC (n=1310):

        median 1.89px,  p25 1.36,  p75 2.78     against sigma = 2

    54% of training pairs are separated by LESS than one sigma and 89% by less
    than two, so the two target Gaussians are nearly the same blob. Training
    almost never asks the network to tell those two channels apart, it learns
    "these corners are the same point" (true 89% of the time), and that prior
    then dominates on the boxes where they genuinely are far apart.

    Dropping sigma is the direct lever on that:

        sigma=2.0 -> 54% of pairs under one sigma   (today)
        sigma=1.5 -> 33%
        sigma=1.0 ->  8%

    1.5 rather than 1.0 because UDP/DARK decoding fits a quadratic to the peak
    curvature for sub-pixel refinement, and a sigma-1.0 blob is only ~3x3
    heatmap pixels -- thin support for that fit. If 1.5 moves collapse without
    hurting the median, 1.0 is the follow-up.

HOW TO JUDGE IT
    NOT on median_px, which this is not expected to move much. The metric
    added for this experiment is in MultiDomainKeypointDistanceMetric.
    Baseline (epoch_120), for comparison:

                                 chewy      ood
        n_collapsed_pairs           10       10
        frac_collapsed_pairs     0.046    0.083
        pair_sep_slope           0.689    0.148     <- 1.0 = faithful
        collapsed_err_share      0.173    0.142
        mean_px                   5.23     5.63
        median_px                 2.67     3.08

    `pair_sep_slope` is the headline: it is the slope of predicted vs true
    corner separation, and it is low because predictions shrink toward the
    typical ~10px gap regardless of the actual box.

SINGLE VARIABLE
    Everything else matches the parent config: Adam (not AdamW), no flips,
    150 epochs, span_weight=0, and absent_weight=1.0 INHERITED.

    absent_weight is deliberately left ON because it is the recipe going to
    production (phantom corners 0.826 -> 0.079 mean peak). That makes the
    correct comparison point the ABSENT run, not epoch_120 -- sigma is the one
    variable relative to `rgbd-geom-finetune-chewy2-absent`. Comparing this
    against epoch_120 instead would confound sigma with absence supervision,
    which is the mistake the flip+span run made.
"""
_base_ = ['./hrnet-w32_8-kp_udp_rgbd_geom_finetune_chewy.py']

# ---------------------------------------------------------------- codec
# A single `model` dict below. Two separate `model = dict(...)` assignments in
# one config silently discard the first -- plain Python rebinding, not a merge.
codec = dict(
    type='UDPHeatmap',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=1.5)               # <- the one variable under test

model = dict(head=dict(decoder=codec))

# The pipelines must be redefined, not inherited. The parent builds its
# GenerateTarget with `encoder={{_base_.codec}}`, which resolves against the
# PARENT's base at parse time -- so redefining `codec` here alone would leave
# the pipeline still encoding at sigma=2 while the decoder used 1.5. That
# mismatch would not raise; it would just quietly produce a broken run.
train_pipeline_sigma = [
    dict(type='LoadRGBDImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomBBoxTransform',
         shift_factor=0.1, scale_factor=[0.75, 1.25], rotate_factor=30),
    dict(type='TopdownAffine', input_size=(256, 256), use_udp=True),
    dict(type='PhotometricDistortionRGBOnly',
         brightness_delta=32, contrast_range=(0.6, 1.4),
         saturation_range=(0.6, 1.4), hue_delta=18),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
val_pipeline_sigma = [
    dict(type='LoadRGBDImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256), use_udp=True),
    dict(type='PackPoseInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline_sigma))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline_sigma))
test_dataloader = dict(dataset=dict(pipeline=val_pipeline_sigma))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SafeMLflowVisBackend',
        exp_name='box-flap-pose',
        run_name='rgbd-geom-finetune-chewy2-sigma15',
        tracking_uri='http://10.200.104.2:31016',
        tags=dict(
            model='hrnet-w32',
            variant='rgbd-geom',
            data='real+chewy2',
            stage='finetune-real+chewy2',
            sigma='1.5',                 # <- the one variable under test
            backbone_init='synthetic-pretrained'),
        # MUST be a tuple: mmengine's scandir() rejects a list.
        artifact_suffix=('.py', '.pth', '.json')),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
