"""STAGE 1 (v2): pretrain on the WIDE synthetic set, unioned with the old one.

WHY A V2 PRETRAIN AT ALL
    The confirmed failure mode is corner-pair collapse driven by regression to
    the mean: `pair_sep_slope` is 0.70 on Chewy and 0.08 on RSC, and collapse
    rate jumps from 7% inside the old synthetic set's p90 to 35-44% outside it.
    Synthetic is 15k boxes against ~430 real, so IT forms the prior. Widening
    the synthetic separation distribution attacks that prior directly, and it
    is the first proposal this session with measured evidence behind it.

WHAT THE NEW SET LOOKS LIKE  (vms_flaps_cropped, 11432 boxes)
                       real (pooled)   old synth   NEW
        median                 0.041       0.066   0.256
        p99                    0.213       0.139   1.316
        max                    0.241       1.060   3.441
        % above 0.139             6%          1%     85%

    The new set OVERSHOOTS: its median box is more extreme than a 1-in-100
    real box. On its own it would swap one mismatched prior for another --
    the old set was uniformly half-open, this one is uniformly wide-open, and
    real boxes are mostly closed with occasional extremes.

WHY THE UNION RATHER THAN THE NEW SET ALONE
    The two sets are complementary, and together they cover what neither does:

        old synth   dense over 0.02 - 0.14   (where real boxes actually live)
        new synth   dense over 0.14 - 3.4    (the tail that was missing)

    The original diagnosis was DENSITY, not reach -- collapse concentrated
    where synthetic support thinned out. A union has support everywhere, which
    is exactly the fix. Set INCLUDE_OLD_SYNTH = False to pretrain on the new
    set alone; everything else in this config is unchanged either way.

    A pretrain does NOT need to match the real prior. It needs to teach the
    model that separation VARIES, and the finetune re-calibrates the location.
    That is the argument for using the wide set despite its median, and it is
    why the finetune stage is unchanged.

SINGLE VARIABLE
    Everything except the training data is identical to
    hrnet-w32_8-kp_udp_rgbd_geom_synth_pretrain.py -- batch 32, 100 epochs,
    cosine, val_interval 10, same optimizer. The ONLY derived change is the
    geometric-loss `warmup_steps`, which is specified in ITERATIONS and so must
    track the dataset size to keep meaning "ramp over the first half".

    Validation stays on the REAL RSC val set, unchanged. The checkpoint worth
    keeping is the one that TRANSFERS, not the one that fits synthetic best.

HOW TO JUDGE IT
    Stage 1 alone proves nothing -- the previous pretrain peaked at
    `best_coco_AP_epoch_70.pth`. The comparison that matters is after the
    finetune, and it is against these numbers:

        chewy pair_sep_slope   0.696      <- the metric this is aimed at
        rsc   pair_sep_slope   0.076      <- the worst domain, most headroom
        chewy median_px        2.667
        max/cyclic_mean_px     6.738 @ep50 (the ship-selection metric)

    A win looks like `pair_sep_slope` moving UP toward 1.0, especially on RSC.
    If slope stays at 0.70/0.08, the synthetic prior was not the mechanism and
    this joins the 0-for-5 list rather than becoming a 1-for-6.

HOW TO LAUNCH
    python tools/train.py \
      configs/hrnet-w32_8-kp_udp_rgbd_geom_synth_pretrain_v2.py \
      --work-dir work_dirs/hrnet_synth_pretrain_v2

    The work-dir name is not free: the stage-2 config's `load_from` points
    into work_dirs/hrnet_synth_pretrain_v2/, so changing one means changing
    both.

    NO max_keep_ckpts, deliberately. Whether zero-shot real accuracy at
    epoch N predicts post-FINETUNE quality has never been checked in this
    project -- `best_coco_AP_epoch_70` was assumed, not verified. The only way
    to find out is to finetune from more than one pretrain checkpoint, and a
    cap of 3 throws the candidates away. 10 files x 329MB is 3.3GB, which is
    cheap against re-running a pretrain to recover one.

    RUN FIRST -- the config trains on the *_trainsplit.json files, which do
    not exist until you make them:
        python scripts/make_synth_val_split.py --n 400 \
          /home/rsquared/Documents/vms_flaps_cropped/annotations/person_keypoints_train.json \
          data/RSC_Keypoints_RGBD/annotations/person_keypoints_synth_only.json

    INCLUDE_OLD_SYNTH cannot be set via --cfg-options: it is read at config
    PARSE time to build the dataset list, and --cfg-options merges into the
    already-built dict. Edit the file to switch it.

PREFLIGHT -- run both before launching, they take under a minute:
    python scripts/preflight_pretrain_set.py <new>/annotations/person_keypoints_train.json
    python scripts/check_synth_coverage.py   <new>/annotations/person_keypoints_train.json --compare-old
"""
_base_ = ['./hrnet-w32_8-kp_udp_rgbd_geom_synth_pretrain.py']

# ---------------------------------------------------------------- what to use
# Union of both synthetic sets. Flip to False for the new set alone.
INCLUDE_OLD_SYNTH = True

# Absolute path: this set lives outside the repo's data/ tree. Edit if it moves.
new_synth_root = '/home/rsquared/Documents/vms_flaps_cropped'
new_synth_ann = 'annotations/person_keypoints_train_trainsplit.json'
new_synth_val = 'annotations/person_keypoints_train_valsplit.json'

old_synth_root = 'data/RSC_Keypoints_RGBD'
old_synth_ann = 'annotations/person_keypoints_synth_only_trainsplit.json'
old_synth_val = 'annotations/person_keypoints_synth_only_valsplit.json'

metainfo_file = 'configs/_base_/datasets/RSC_Keypoints.py'

# The pretrain chain registers the custom transforms, preprocessor, loss and
# MLflow backend, but NOT the multi-domain metrics -- those were only ever
# pulled in by the chewy finetune config. Stage 1 now uses them too, and an
# unregistered metric fails at build_val_loop, i.e. AFTER the model is built
# and an MLflow run has been opened.
custom_imports = dict(
    imports=[
        'mmpose.datasets.transforms.rgbd',
        'mmpose.models.data_preprocessors.nchannel',
        'mmpose.models.losses.geometric_loss',
        'mmpose.engine.vis_backends.safe_mlflow',
        'mmpose.evaluation.metrics.multi_domain_coco_metric',
        'mmpose.evaluation.metrics.multi_domain_distance_metric',
    ],
    allow_failed_imports=False)

# POST-SPLIT annotation counts (400 images held out of each by
# scripts/make_synth_val_split.py --n 400). Used only to size the loss warmup
# below; an out-of-date number changes when the geometric loss ramps in, which
# is a silent training change.
N_NEW = 11432 - 400
N_OLD = 18530 - 400

# Real held-out sets. Stage 1 trains on ZERO real images, so all three are
# legitimately held out here -- a luxury stage 2 does not have.
rsc_root = 'data/RSC_Keypoints_RGBD'
chewy_root = 'data/ChewyCrops'
ood_root = 'data/OOD'

# ------------------------------------------------------------------ pipelines
# Redefined in full because `dataset` is replaced wholesale (CocoDataset ->
# CombinedDataset needs _delete_), which drops the pipeline the base supplied.
# Identical to the base RGBD pipeline -- do not let it drift.
train_pipeline = [
    dict(type='LoadRGBDImage'),
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


def _subset(root, ann, test_mode=False):
    d = dict(
        type='CocoDataset',
        data_root=root,
        ann_file=ann,
        data_prefix=dict(img='images/'),
        metainfo=dict(from_file=metainfo_file),
        pipeline=[])          # CombinedDataset owns the pipeline
    if test_mode:
        d['test_mode'] = True
    return d


_datasets = [_subset(new_synth_root, new_synth_ann)]
if INCLUDE_OLD_SYNTH:
    _datasets.append(_subset(old_synth_root, old_synth_ann))

BATCH_SIZE = 32
_n_samples = N_NEW + (N_OLD if INCLUDE_OLD_SYNTH else 0)
_iters_per_epoch = _n_samples // BATCH_SIZE

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=8,
    dataset=dict(
        _delete_=True,
        type='CombinedDataset',
        metainfo=dict(from_file=metainfo_file),
        datasets=_datasets,
        # [new, old] in natural proportion -- 11432 : 18530, so the wide set is
        # 38% of each batch. Raise the first factor to weight the wide tail
        # harder; that is the one lever here worth sweeping if the first run
        # moves `pair_sep_slope` but not far enough.
        sample_ratio_factor=[1.0, 1.0] if INCLUDE_OLD_SYNTH else [1.0],
        pipeline=train_pipeline))

# Geometric loss ramps 0 -> 1 over the first HALF of training, as before. The
# base config's 15650 was 313 iters/epoch x 50; that iters/epoch no longer
# holds, so it is recomputed rather than inherited.
model = dict(head=dict(loss=dict(warmup_steps=_iters_per_epoch * 50)))

# ------------------------------------------------------------------- tracking
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SafeMLflowVisBackend',
        exp_name='box-flap-pose',
        run_name=('rgbd-geom-synth-pretrain-v2-union' if INCLUDE_OLD_SYNTH
                  else 'rgbd-geom-synth-pretrain-v2-widenew'),
        tracking_uri='http://10.200.104.2:31016',
        tags=dict(
            model='hrnet-w32',
            variant='rgbd-geom',
            data=('synth-union-wide+old' if INCLUDE_OLD_SYNTH
                  else 'synth-wide-only'),
            train_size=str(_n_samples),
            stage='pretrain',
            backbone_init='imagenet'),
        # MUST be a tuple: mmengine's scandir() rejects a list.
        artifact_suffix=('.py', '.pth', '.json')),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')


# ------------------------------------------------------------------ validation
# The pretrain is judged on GENERALISATION, so it validates on every real set
# rather than the 72-image RSC val alone:
#
#     RSC val     72 imgs / 180 pairs
#     Chewy val   76 imgs / 128 pairs
#     OOD         30 imgs / 120 pairs   <- a different capture session
#     -------------------------------
#                178 imgs / 428 pairs   (was 72 / 180)
#
# Plus a held-out slice of EACH synthetic set, which is what separates "still
# converging" from "memorising the generator": if synthetic val keeps improving
# while the real sets go backwards, stop.
#
# The synthetic domains carry `select=False`, so they are reported but can
# never drive save_best. Selecting a pretrain on synthetic accuracy would
# reward exactly the overfitting this split exists to detect.
#
# ROUTING SUBTLETY: domains are matched by img_path prefix, longest root first
# (MultiDomain*Metric._route). The old synthetic images live UNDER the real RSC
# tree at data/RSC_Keypoints_RGBD/images/synthetic/, so `synth_old` is routed
# on that deeper path while `rsc` keeps the shallow root and picks up the rest.
# The DATASET below still uses the plain rsc_root -- only the metric's routing
# key is the deeper path.
old_synth_route = f'{old_synth_root}/images/synthetic'

val_pipeline = [
    dict(type='LoadRGBDImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256), use_udp=True),
    dict(type='PackPoseInputs'),
]

# Shared by val and test. test_mode=True is what every other evaluation set in
# this project uses; it was missing from the completed stage-1 run, which
# retained all images anyway (72/30/400 confirmed by the routing histogram),
# so the epoch-30 selection is unaffected.
_eval_subsets = [
    _subset(rsc_root, 'annotations/person_keypoints_val.json', test_mode=True),
    _subset(chewy_root, 'annotations/person_keypoints_Train.json', test_mode=True),
    _subset(ood_root, 'annotations/person_keypoints_Test.json', test_mode=True),
    _subset(new_synth_root, new_synth_val, test_mode=True),
    _subset(old_synth_root, old_synth_val, test_mode=True),
]

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='CombinedDataset',
        metainfo=dict(from_file=metainfo_file),
        datasets=_eval_subsets,
        pipeline=val_pipeline))

_val_domains = [
    dict(name='rsc', data_root=rsc_root),
    dict(name='chewy', data_root=chewy_root),
    dict(name='ood', data_root=ood_root),
    dict(name='synth_new', data_root=new_synth_root, select=False),
    dict(name='synth_old', data_root=old_synth_route, select=False),
]
_val_domains_coco = [
    dict(name='rsc', data_root=rsc_root,
         ann_file=f'{rsc_root}/annotations/person_keypoints_val.json'),
    dict(name='chewy', data_root=chewy_root,
         ann_file=f'{chewy_root}/annotations/person_keypoints_Train.json'),
    dict(name='ood', data_root=ood_root,
         ann_file=f'{ood_root}/annotations/person_keypoints_Test.json'),
    dict(name='synth_new', data_root=new_synth_root, select=False,
         ann_file=f'{new_synth_root}/{new_synth_val}'),
    dict(name='synth_old', data_root=old_synth_route, select=False,
         ann_file=f'{old_synth_root}/{old_synth_val}'),
]

val_evaluator = [
    dict(type='MultiDomainKeypointDistanceMetric', domains=_val_domains),
    dict(type='MultiDomainCocoMetric', domains=_val_domains_coco),
]

# `_delete_` is required at the TOP level of these two because the base sets
# them to None. It must NOT appear on the nested `dataset`: mmengine strips
# `_delete_` while merging a child into a base dict, and with a base of None
# there is no merge to strip it, so it survives into CombinedDataset.__init__
# and raises `unexpected keyword argument '_delete_'`. val_dataloader gets
# away with the nested key precisely because it DOES have a base to merge
# into. It must also not appear inside test_evaluator, which is a list whose
# elements are constructed rather than merged.
#
# Nothing is inherited here either, so the sampler and loader flags have to be
# spelled out -- copied from the chewy finetune config, which is the version
# proven to work under tools/test.py.
test_cfg = dict(_delete_=True)
test_dataloader = dict(
    _delete_=True,
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file=metainfo_file),
        datasets=_eval_subsets,
        pipeline=val_pipeline))
test_evaluator = val_evaluator

# Two selection criteria, deliberately. `min/coco/AP` is the historical
# continuity metric; `max/cyclic_mean_px` is the one the finetune ships on, so
# a pretrain that already looks good there is the better bet. Both are floors
# ("is this checkpoint broken"), not fine-grained selectors -- see the note on
# max_keep_ckpts above for why the periodic checkpoints are kept too.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best=['min/coco/AP', 'max/cyclic_mean_px'],
        rule=['greater', 'less']))
