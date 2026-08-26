"""TEST ARM: drop the cardinal names, detect at most 8 corners.

NOT A PRODUCTION PATH, and nothing else in the repo imports this. It exists to
measure ONE number: how much localisation accuracy the cardinal formulation
costs.

THE CHANGE
    The production model spends 8 heatmap channels on 8 NAMED corners, so it
    must decide both WHERE each corner is and WHICH one it is. This arm
    renders all of them into ONE channel and decodes the peaks back as an
    unordered set -- "at most 8 keypoints", no identity. Scoring matches
    predictions to ground truth optimally, so any permutation is free.

    The gap between this arm's `mean_px` and the cardinal model's is the
    price of naming.

WHAT IS DELIBERATELY DIFFERENT, AND WHY
    codec       AgnosticUDPHeatmap -- 8 gaussians MAX-reduced to one map.
    head        out_channels=1 instead of 8.
    loss        plain KeypointMSELoss. The geometric loss constrains angles
                between NAMED edges (adjacent flaps perpendicular, opposite
                ones parallel) and cannot be expressed without identity. This
                is a real confound: this arm loses the geometric term as well
                as the naming, so a win here is not attributable to the naming
                alone. Run the cardinal baseline with perp_weight=0 if the
                result is interesting enough to need the clean attribution.
    evaluator   AgnosticKeypointDistanceMetric (Hungarian). CocoMetric is gone
                -- OKS is defined per named keypoint and is meaningless here.

WHAT IS DELIBERATELY THE SAME
    Data, splits, augmentation, optimizer, schedule, domains and `load_from`
    all come from the cardinal chewy finetune, so the comparison is clean on
    everything except the two changes above.

    `load_from` will report a shape mismatch on the head's final layer (8 vs 1
    output channel) and skip it. That is expected -- the backbone transfers,
    the head is new.

HOW TO READ IT
    Compare `rsc/mean_px` and `chewy/mean_px` against the cardinal run's
    `cyclic_mean_px` (the rotation-tolerant version -- the fair comparison,
    since this arm gets permutations for free):

        cardinal baseline    rsc 8.64    chewy 5.70

    Also watch `n_missing` and `n_spurious`. The cardinal model always emits
    exactly 8 points, so it cannot miss one; this arm can, and a large
    n_missing means the accuracy gain is partly bought by not answering.

    `corner_span_err_px` is computed on the labelling INDUCED by the matching,
    so it is directly comparable to the cardinal metric's. Baseline: rsc 2.61,
    chewy 2.27.

LAUNCH
    python tools/train.py configs/hrnet-w32_8-kp_udp_rgbd_agnostic.py \\
      --work-dir work_dirs/hrnet_agnostic_test
"""
_base_ = ['./hrnet-w32_8-kp_udp_rgbd_geom_finetune_chewy.py']

custom_imports = dict(
    imports=[
        'mmpose.datasets.transforms.rgbd',
        'mmpose.models.data_preprocessors.nchannel',
        'mmpose.models.losses.geometric_loss',
        'mmpose.engine.vis_backends.safe_mlflow',
        'mmpose.codecs.agnostic_udp_heatmap',
        'mmpose.evaluation.metrics.agnostic_distance_metric',
    ],
    allow_failed_imports=False)

metainfo_file = 'configs/_base_/datasets/RSC_Keypoints.py'
rsc_root = 'data/RSC_Keypoints_RGBD'
chewy_root = 'data/ChewyCrops2'
chewy_val_root = 'data/ChewyCrops'

codec = dict(
    type='AgnosticUDPHeatmap',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2,
    max_keypoints=8,
    score_thr=0.1,
    nms_kernel=5)

model = dict(
    head=dict(
        out_channels=1,
        decoder=codec,
        loss=dict(
            _delete_=True,
            type='KeypointMSELoss',
            use_target_weight=True)))

# Pipelines are redefined IN FULL rather than patched: `{{_base_.codec}}` in a
# parent resolves against the PARENT's base at parse time, so overriding
# `codec` here would not reach the inherited pipeline and the encoder would
# silently stay the 8-channel one while the head decoded with the new codec.
train_pipeline = [
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
val_pipeline = [
    dict(type='LoadRGBDImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256), use_udp=True),
    dict(type='PackPoseInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))

_domains = [
    dict(name='rsc', data_root=rsc_root),
    dict(name='chewy', data_root=chewy_val_root),
]
val_evaluator = [
    dict(type='AgnosticKeypointDistanceMetric', domains=_domains,
         fail_thr_px=20.0),
]
test_evaluator = val_evaluator
test_dataloader = dict(dataset=dict(pipeline=val_pipeline))

# The inherited criteria (`max/cyclic_mean_px`) do not exist on this metric.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best=['max/mean_px', 'max/corner_span_err_px'],
        rule='less'))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SafeMLflowVisBackend',
        exp_name='box-flap-pose',
        run_name='rgbd-agnostic-corners-test',
        tracking_uri='http://10.200.104.2:31016',
        tags=dict(model='hrnet-w32', variant='agnostic-corners',
                  data='chewy2+rsc', stage='test'),
        artifact_suffix=('.py', '.pth', '.json')),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
