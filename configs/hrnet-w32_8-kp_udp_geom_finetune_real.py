"""STAGE 2 (RGB arm): fine-tune on real colour, from the synthetic-pretrained
checkpoint produced by stage 1.

Mirror of hrnet-w32_8-kp_udp_rgbd_geom_finetune_real.py with depth removed.
LR, paramwise backbone multiplier, epoch budget, geometric warmup and schedule
are copied verbatim so the RGB and RGBD arms differ only in input modality —
which makes the existing RGBD run 270a8336 (best coco/AP 0.9446) a valid
control for this one.

IMPORTANT: after stage 1 finishes, point `load_from` below at its actual best
checkpoint (e.g. best_coco_AP_epoch_XX.pth). The path below is a placeholder.
"""
_base_ = ['./hrnet-w32_8-kp_udp_geom.py']

custom_imports = dict(
    imports=[
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
        run_name='rgb-geom-finetune-real',
        tracking_uri='http://10.200.104.2:31016',
        tags=dict(
            model='hrnet-w32',
            variant='rgb-geom',          # <- the one difference from the RGBD arm
            data='real-only',
            train_size='288',
            stage='finetune-from-synth',
            backbone_init='synthetic-pretrained'),
        artifact_suffix=['.py', '.pth', '.json']),
]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# PLACEHOLDER — update to stage 1's best checkpoint before running.
load_from = 'work_dirs/hrnet_rgb_synth_pretrain/best_coco_AP_epoch_100.pth'
resume = False

# Real data only — same 288 train / 72 val split used everywhere else.
train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/person_keypoints_train.json'))

# Fine-tuning: lower LR than stage 1, backbone slower still to preserve the
# features learned from synthetic pretraining.
optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys=dict(
            backbone=dict(lr_mult=0.5),
        )))

train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=10)

# Heatmaps are already well-formed from stage 1, so the geometric prior can
# activate quickly. 200 steps ~ 11 epochs at 288/batch=16.
model = dict(
    head=dict(
        loss=dict(warmup_steps=200)))

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False,
         begin=0, end=50),
    dict(type='CosineAnnealingLR', eta_min=0, begin=0, end=150,
         by_epoch=True),
]
