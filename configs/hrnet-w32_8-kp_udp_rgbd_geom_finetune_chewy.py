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
            train_size='288+70(x2)',
            stage='finetune-real+chewy2',
            backbone_init='synthetic-pretrained'),
        artifact_suffix=['.py', '.pth', '.json']),
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
chewy_root = 'data/ChewyCrops2'

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
        # [RSC, Chewy] -- oversample the 70 new images 2x -> ~33% of the batch.
        # Lower than it would be for a short continuation run: over 150 epochs
        # a 3x factor shows each Chewy image ~450 times, which overfits 70
        # images. Raise it if the Chewy pre-labels come out weak, lower it
        # toward 1.0 if val AP sags.
        sample_ratio_factor=[1.0, 2.0],
        pipeline=train_pipeline))

# Validation stays on the ORIGINAL real val split, and a plain CocoDataset,
# because CocoMetric needs a single ann_file. It does NOT measure Chewy
# performance — nothing does, since all 70 labelled Chewy images are in train.
# Its job is to catch forgetting: if coco/AP falls well below the ~0.94
# stage-2 plateau (~0.94), something is wrong. It should land in the same
# range as stage 2, since this IS stage 2 with extra data.

# ---------------------------------------------------------------- schedule
# Stage 2's schedule verbatim — starting from the synthetic pretrain, the model
# has to actually learn the real domain, not just nudge toward a new rig.
optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys=dict(backbone=dict(lr_mult=0.5))))

# 288 + 140 = 428 effective samples / batch 16 = 27 iters per epoch.
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=10)

# Same as stage 2: heatmaps from the synthetic pretrain are already well-formed,
# so the geometric prior can engage quickly. 200 steps ~ 7 epochs here.
model = dict(head=dict(loss=dict(warmup_steps=200)))

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False,
         begin=0, end=50),
    dict(type='CosineAnnealingLR', eta_min=0, begin=0, end=150, by_epoch=True),
]
