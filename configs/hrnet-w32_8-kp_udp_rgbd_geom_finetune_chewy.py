"""STAGE 2 (re-run): finetune the synthetic-pretrained RGBD model on ALL real
data — the original RSC set plus the new Chewy customer rig.

This deliberately restarts from the **synthetic pretrain**, not from the
stage-2 finetuned checkpoint. Chaining finetune-on-finetune would work, but:

  * It is the SAME recipe already validated twice (0.9446 / 0.9433), just with
    more real data. Continuing from a finetuned model would be a new,
    unvalidated 3-stage schedule.
  * The model learns both rigs JOINTLY instead of learning rig A and then being
    nudged toward rig B, which is what invites catastrophic forgetting.
  * It scales. This loop repeats every time a customer set is labelled; a
    chain of finetunes accumulates path dependence and becomes impossible to
    reproduce, whereas "synth-pretrain -> finetune on all real data" is a
    fixed, repeatable procedure no matter how many sets exist.

Because it restarts from the pretrain, the schedule matches stage 2 (lr 1e-4,
150 epochs) rather than the short low-LR schedule a continuation would use.

DATASET LAYOUT
    Each collected set stays in its own self-contained folder:

        <anywhere>/rsc_rgbd/{images,depth,annotations}
        <anywhere>/ChewyCrops2/{images,depth,annotations}
        <anywhere>/<future_set>/{images,depth,annotations}

    and `CombinedDataset` unions them at config level. No symlinks, no merged
    annotation file, no rewriting of file_name or ids — each set keeps its
    original annotations untouched, which matters for customer data you may
    have to hand back or re-export.

    Adding a future set = one more entry in `datasets` below.

    `sample_ratio_factor` re-weights the sets. 70 Chewy images against 288 RSC
    is only 20% of the batch; the 3.0 below oversamples Chewy to ~210 so the
    new rig is roughly 42% of what the model sees. Lower it toward 1.0 if the
    model starts forgetting the original rig (watch coco/AP on val).

    NOTE sub-datasets carry `pipeline=[]` — CombinedDataset owns the shared
    pipeline and applies it after the sub-dataset returns raw data_info.

Purpose is a *pre-labelling* model: good enough that correcting the remaining
316 images in CVAT is fast. Not a deployment candidate — for that, gate on the
OOD set, not on val (see the RGB-vs-RGBD result where val ranked them backwards).

Note on the new data: 34 of the 70 images are only partially labelled — the
SOUTH flap is often not visible from this rig, so EAST_2/SOUTH_1/SOUTH_2/WEST_1
are frequently v=0. Those points are excluded from the loss via
keypoint_weights, so partial labels cost supervision but do no harm.
"""
_base_ = ['./hrnet-w32_8-kp_udp_rgbd_geom.py']

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

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SafeMLflowVisBackend',
        exp_name='box-flap-pose',
        run_name='rgbd-geom-finetune-chewy2',
        tracking_uri='http://10.200.104.2:31016',
        tags=dict(
            model='hrnet-w32',
            variant='rgbd-geom',
            data='real+chewy2',
            train_size='288+140(x1)',
            stage='finetune-real+chewy2',
            backbone_init='synthetic-pretrained'),
        # MUST be a tuple: mmengine's scandir() rejects a list with
        # '"suffix" must be a string or tuple of strings'. MLflowVisBackend
        # calls it inside close(), so a list makes close() raise — which
        # aborts artifact upload (this is why .pth files never reached MLflow)
        # and skips mlflow.end_run(), leaving runs stuck in RUNNING.
        artifact_suffix=('.py', '.pth', '.json')),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# The SYNTHETIC-pretrained backbone — the same starting point stage 2 used.
# This is what hrnet-w32_8-kp_udp_rgbd_geom_finetune_real.py loads.
# (run 270a8336, the best stage 2 at AP 0.9446, instead started from
#  work_dirs/hrnet_synth_pretrain_test_01/best_coco_AP_epoch_20.pth)
load_from = 'work_dirs/hrnet_synth_pretrain_test_02/best_coco_AP_epoch_70.pth'
resume = False

# ---------------------------------------------------------------- datasets
metainfo_file = 'configs/_base_/datasets/RSC_Keypoints.py'

# Point these at wherever each set lives on the training box.
rsc_root = 'data/RSC_Keypoints_RGBD'
chewy_root = 'data/ChewyCrops2'          # training set, captured 2026-08-18
chewy_val_root = 'data/ChewyCrops'       # val set,      captured 2026-08-13

train_pipeline = [
    dict(type='LoadRGBDImage'),
    dict(type='GetBBoxCenterScale'),
    # NOTE: RandomFlip / RandomFlipVertical were tested together with AdamW
    # and a 100-epoch schedule and the bundle REGRESSED chewy median from
    # 2.38 to 2.89px. RandomFlipVertical still exists in
    # mmpose/datasets/transforms/rgbd.py and is verified correct -- retest it
    # on its own, not bundled, before re-enabling.
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


train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='CombinedDataset',
        metainfo=dict(from_file=metainfo_file),
        datasets=[
            _subset(rsc_root, 'annotations/person_keypoints_train.json'),
            _subset(chewy_root, 'annotations/person_keypoints_Train.json'),
        ],
        # [RSC, Chewy]. The factor exists to hold Chewy's SHARE of each batch
        # roughly constant as the set grows -- so it must come down every time
        # more Chewy images are labelled, or the balance silently drifts:
        #     70 imgs x 2.0 = 140 effective -> 32.7% of the batch  (round 1)
        #    140 imgs x 1.0 = 140 effective -> 32.7% of the batch  (round 2, here)
        #    140 imgs x 2.0 = 280 effective -> 49.3%  <- what leaving it at 2.0 does
        # 32.7% is the ratio that produced val 0.9491 / OOD PCK 97.5%, so keep
        # it there and let the DATA be the only thing that changed.
        # Rule of thumb for the next round: factor ~= 140 / n_chewy_images.
        sample_ratio_factor=[1.0, 1.0],
        pipeline=train_pipeline))

# ---------------------------------------------------------------- validation
# BOTH RSC and Chewy are deployment targets, so the model is selected on
# whichever it is WORSE at -- `min/coco/AP` -- rather than on either alone or
# on their average. A mean lets a checkpoint buy Chewy accuracy with RSC
# accuracy and still look good; the minimum only improves when the weaker
# domain improves.
#
# The Chewy val is a held-out SESSION (captured 2026-08-13), not held-out
# images from the training session (2026-08-18) -- zero filename overlap with
# the 140 training images -- so it measures generalisation within the rig
# rather than memorisation.
val_pipeline = [
    dict(type='LoadRGBDImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256), use_udp=True),
    dict(type='PackPoseInputs'),
]

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='CombinedDataset',
        metainfo=dict(from_file=metainfo_file),
        datasets=[
            dict(type='CocoDataset', data_root=rsc_root,
                 ann_file='annotations/person_keypoints_val.json',
                 data_prefix=dict(img='images/'),
                 metainfo=dict(from_file=metainfo_file),
                 test_mode=True, pipeline=[]),
            dict(type='CocoDataset', data_root=chewy_val_root,
                 ann_file='annotations/person_keypoints_Train.json',
                 data_prefix=dict(img='images/'),
                 metainfo=dict(from_file=metainfo_file),
                 test_mode=True, pipeline=[]),
        ],
        pipeline=val_pipeline))

# Two evaluators run together. Selection is on RAW PIXEL ERROR; coco/AP is
# kept only for continuity with the historical runs.
#
# Why not AP: this project rewrites bboxes to full-image, so OKS takes
# s = sqrt(image area) (296-725 px on the Chewy val). Even the strictest
# threshold in AP@0.50:0.95 then tolerates ~33 px, while the model's median
# error is ~3 px — AP saturates near 1.0 and cannot separate checkpoints.
# Observed directly in run 21677dc6: chewy/coco/AP sat at 0.988-0.998 across
# every epoch while rsc/coco/AP moved normally, so `min` was pinned to RSC and
# the Chewy signal contributed nothing.
#
# Why not a size-normalised error: the model emits pixels and downstream
# converts to physical units itself using the depth map, so dividing by image
# size double-corrects. Dividing by crop diagonal would also over-correct —
# crop size spans ~2.5x here while flap height (216-586 mm) implies far less
# variation in true camera-to-corner distance.
#
# Samples are routed to a domain by img_path against data_root, NOT img_id:
# the two annotation files both number from 1 and would otherwise collide.
val_evaluator = [
    dict(
        # NOTE: no _delete_ here. It is consumed by mmengine's *dict* merge;
        # inside a list element it survives and reaches the constructor as an
        # unexpected kwarg. Assigning a list to val_evaluator already replaces
        # the base's dict wholesale, so no _delete_ is needed.
        type='MultiDomainKeypointDistanceMetric',
        domains=[
            dict(name='rsc', data_root=rsc_root),
            dict(name='chewy', data_root=chewy_val_root),
        ],
        # a corner off by this much is unusable for span metrology
        fail_thr_px=20.0),
    dict(
        type='MultiDomainCocoMetric',
        domains=[
            dict(name='rsc', data_root=rsc_root,
                 ann_file=f'{rsc_root}/annotations/person_keypoints_val.json'),
            dict(name='chewy', data_root=chewy_val_root,
                 ann_file=f'{chewy_val_root}/annotations/person_keypoints_Train.json'),
        ]),
]

# NOTE the criterion INVERTS versus an AP-style metric: with an error metric,
# "works everywhere" means minimise the MAXIMUM — hence max/mean_px + 'less'.
# `best_max_mean_px_epoch_N.pth` is the one to ship.
#
# mean_px drives selection rather than median deliberately: for corner-to-corner
# metrology a large error does not degrade a measurement, it destroys it, and a
# median shrugs those off. median_px is logged alongside so you can tell WHY
# mean moved — mean up with median flat means new catastrophic failures; both
# up means broad degradation.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        # Selection is on the ROTATION-TOLERANT error. Downstream associates
        # the four spans across the two camera views, so a cyclically clocked
        # flap assignment (NORTH read as EAST etc, common on rotated boxes)
        # costs nothing in the product -- but it moves every corner about one
        # box-side, which makes plain mean_px enormous for a prediction whose
        # geometry is correct. Selecting on mean_px would chase a labelling
        # convention instead of accuracy.
        # `best_max_cyclic_mean_px_epoch_N.pth` is the one to ship.
        save_best=['max/cyclic_mean_px',
                   'rsc/cyclic_mean_px', 'chewy/cyclic_mean_px'],
        rule='less'))

# ---------------------------------------------------------------- test
# Mirrors val exactly, so `tools/test.py <config> <ckpt>` evaluates a chosen
# checkpoint on BOTH domains in one pass — including the worst-offender report
# that names the file and keypoint behind each large error.
#
# `_delete_` is needed on these two because the base sets them to None and a
# dict-valued child is merged; it must NOT appear inside `test_evaluator`,
# which is a list whose elements are constructed rather than merged.
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
        datasets=[
            dict(type='CocoDataset', data_root=rsc_root,
                 ann_file='annotations/person_keypoints_val.json',
                 data_prefix=dict(img='images/'),
                 metainfo=dict(from_file=metainfo_file),
                 test_mode=True, pipeline=[]),
            dict(type='CocoDataset', data_root=chewy_val_root,
                 ann_file='annotations/person_keypoints_Train.json',
                 data_prefix=dict(img='images/'),
                 metainfo=dict(from_file=metainfo_file),
                 test_mode=True, pipeline=[]),
        ],
        pipeline=val_pipeline))

test_evaluator = [
    dict(type='MultiDomainKeypointDistanceMetric',
         domains=[dict(name='rsc', data_root=rsc_root),
                  dict(name='chewy', data_root=chewy_val_root)],
         fail_thr_px=20.0,
         # more offenders than during training: this is the audit pass
         report_worst=15),
    dict(type='MultiDomainCocoMetric',
         domains=[
             dict(name='rsc', data_root=rsc_root,
                  ann_file=f'{rsc_root}/annotations/person_keypoints_val.json'),
             dict(name='chewy', data_root=chewy_val_root,
                  ann_file=f'{chewy_val_root}/annotations/person_keypoints_Train.json'),
         ]),
]

# ---------------------------------------------------------------- schedule
# Stage 2's schedule verbatim — starting from the synthetic pretrain, the model
# has to actually learn the real domain, not just nudge toward a new rig.
# Baseline optimiser. AdamW + weight_decay=1e-4 was tried as part of the
# flip+span bundle that regressed; reverted so absent_weight is the only
# variable under test here.
optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys=dict(backbone=dict(lr_mult=0.5))))

# 288 + 140 = 428 effective samples / batch 16 = 27 iters per epoch.
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=10)

# Same as stage 2: heatmaps from the synthetic pretrain are already well-formed,
# so the geometric prior can engage quickly. 200 steps ~ 7 epochs here.
# The span term rides the same warmup ramp -- it reads coordinates out of the
# heatmaps via soft-argmax, so it is meaningless until the peaks are sharp.
model = dict(
    head=dict(
        loss=dict(
            warmup_steps=200,
            # Flap length is what downstream measures and what neither angular
            # term can see. Weight is the one knob here that wants a sweep:
            # the angular terms sit at 0.005 but are near-zero in practice
            # (their free zones are rarely violated), whereas this term fires
            # on the ~2.5% typical span error, so a comparable weight would be
            # far too weak. 0.1 puts it at roughly a third of the MSE term.
            # The span term was measured NOT to work: it optimises soft-argmax
            # coordinates while inference decodes with argmax+DARK, so the
            # along-edge error it targets moved 4.08 -> 4.07px. Left off.
            span_weight=0.0,
            span_tolerance=0.01,
            # Absent-keypoint suppression. The codec already emits an all-zero
            # target for a v=0 keypoint; the weight of 0 was throwing that
            # supervision away, leaving the channel with exactly zero gradient.
            # Measured on ChewyCrops: 169 corners that are NOT in the image
            # still peak at a mean of 0.83 (70% above 0.8), indistinguishable
            # from the 0.96 of real detections -- so nothing downstream can
            # filter them. 1.0 means "train this channel like any other";
            # there is no magic constant to tune here.
            absent_weight=1.0)))

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False,
         begin=0, end=50),
    dict(type='CosineAnnealingLR', eta_min=0, begin=0, end=150, by_epoch=True),
]
