_base_ = ['./hrnet-w32_8-kp_udp_rgbd_geom.py']

# STAGE 1 of two-stage training: pretrain on ~10K synthetic samples.
#
# At this scale, the model won't just memorize the synthetic set — it should
# learn genuinely transferable features. Training dynamics change vs the
# 168-sample first attempt: longer warmup, more epochs before overfitting.

# Register custom modules + MLflow
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
        run_name='rgbd-geom-synth-pretrain-10k',
        tracking_uri='http://10.200.104.2:31016',
        tags=dict(
            model='hrnet-w32',
            variant='rgbd-geom',
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

# Train on synthetic only. Assumes the ~10K annotation file is at
# annotations/person_keypoints_synth_only.json and the images/depth are
# symlinked under images/synthetic/ and depth/synthetic/ (same layout as before).
train_dataloader = dict(
    batch_size=32,           # 2x default — 10K samples benefits from bigger batches
    num_workers=8,           # more workers to keep GPU fed
    dataset=dict(
        ann_file='annotations/person_keypoints_synth_only.json'))

# At 10K/batch 32 = 313 iters per epoch. 100 epochs = 31,300 iters total.
# Model shouldn't overfit at this data scale until much later than 168-sample
# case (which peaked at epoch 20). Give it room to learn.
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=10)

# Geometric-loss warmup at the new scale:
#   313 iters/epoch × 50 epochs = 15,650 warmup steps
# Ramps geometric weight from 0 -> 1 over the first half of training.
model = dict(
    head=dict(
        loss=dict(warmup_steps=15650)))

# Cosine schedule matched to the new epoch count.
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False,
         begin=0, end=1000),          # ~3 epochs of LR warmup
    dict(type='CosineAnnealingLR', eta_min=0, begin=0, end=100,
         by_epoch=True),
]
