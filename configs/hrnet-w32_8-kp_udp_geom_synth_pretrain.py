"""STAGE 1 (RGB arm): pretrain on ~10K synthetic colour images.

Mirror of hrnet-w32_8-kp_udp_rgbd_geom_synth_pretrain.py with depth removed.
Batch size, worker count, epoch budget, LR schedule and geometric warmup are
copied verbatim so the RGB and RGBD arms differ only in input modality.

Reads the same annotation file as the RGBD arm
(annotations/person_keypoints_synth_only.json) from the same data root, and
simply never touches depth/.
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
        run_name='rgb-geom-synth-pretrain-10k',
        tracking_uri='http://10.200.104.2:31016',
        tags=dict(
            model='hrnet-w32',
            variant='rgb-geom',          # <- the one difference from the RGBD arm
            data='synthetic-only',
            train_size='10000',
            stage='pretrain',
            backbone_init='imagenet'),
        artifact_suffix=['.py', '.pth', '.json']),
]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# Train on synthetic only.
train_dataloader = dict(
    batch_size=32,           # 2x default — 10K samples benefits from bigger batches
    num_workers=8,
    dataset=dict(
        ann_file='annotations/person_keypoints_synth_only.json'))

# 10K/batch 32 = 313 iters per epoch; 100 epochs = 31,300 iters.
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=10)

# 313 iters/epoch x 50 epochs = 15,650 steps: geometric weight ramps from
# 0 -> 1 over the first half of training.
model = dict(
    head=dict(
        loss=dict(warmup_steps=15650)))

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False,
         begin=0, end=1000),          # ~3 epochs of LR warmup
    dict(type='CosineAnnealingLR', eta_min=0, begin=0, end=100,
         by_epoch=True),
]
