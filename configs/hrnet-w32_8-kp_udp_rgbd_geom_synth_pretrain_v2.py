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
      --work-dir work_dirs/hrnet_synth_pretrain_v2 \
      --cfg-options default_hooks.checkpoint.max_keep_ckpts=3

    The work-dir name is not free: the stage-2 config's `load_from` points
    into work_dirs/hrnet_synth_pretrain_v2/, so changing one means changing
    both.

    Capping periodic checkpoints is right for STAGE 1 and wrong for stage 2.
    Here the only checkpoint that matters is the one that transfers, and
    sweeping pretrain epochs post-hoc would cost a full finetune each. At
    interval 10 over 100 epochs that is 10 files x 329MB (they carry optimizer
    state; the best_* copies are ~110MB), so the cap saves ~2GB for nothing
    lost. Do NOT carry the flag over to the finetune -- see that config.

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
new_synth_ann = 'annotations/person_keypoints_train.json'

old_synth_root = 'data/RSC_Keypoints_RGBD'
old_synth_ann = 'annotations/person_keypoints_synth_only.json'

metainfo_file = 'configs/_base_/datasets/RSC_Keypoints.py'

# Annotation counts, used only to size the loss warmup below. If the new set
# grows, update N_NEW -- an out-of-date number changes when the geometric loss
# ramps in, which is a silent training change.
N_NEW = 11432
N_OLD = 18530

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


def _subset(root, ann):
    return dict(
        type='CocoDataset',
        data_root=root,
        ann_file=ann,
        data_prefix=dict(img='images/'),
        metainfo=dict(from_file=metainfo_file),
        pipeline=[])          # CombinedDataset owns the pipeline


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
