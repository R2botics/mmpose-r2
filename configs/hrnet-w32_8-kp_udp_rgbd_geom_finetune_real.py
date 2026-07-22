_base_ = ['./hrnet-w32_8-kp_udp_rgbd_geom.py']

# STAGE 2 of two-stage training: fine-tune on real, starting from the
# synthetic-pretrained checkpoint produced by stage 1.
#
# IMPORTANT: after stage 1 completes, update `load_from` below to point at
# the actual best checkpoint from stage 1 (e.g., best_coco_AP_epoch_XX.pth).

# Custom modules + MLflow
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
        run_name='rgbd-geom-finetune-real',
        tracking_uri='http://10.200.104.2:31016',
        tags=dict(
            model='hrnet-w32',
            variant='rgbd-geom',
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

# Load the full model (backbone + head + preprocessor buffers) from stage 1.
# Update this path after stage 1 completes and you know its best epoch.
load_from = 'work_dirs/hrnet_synth_pretrain_test_01/best_coco_AP_epoch_20.pth'
resume = False

# Train on real data only (same 288 train / 72 val split used everywhere else).
train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/person_keypoints_train.json'))

# Fine-tuning: lower LR than stage 1 (which used 5e-4), and shorter schedule
# because the model already knows the task.
optim_wrapper = dict(
    optimizer=dict(lr=1e-4),                  # was 5e-4 in base config
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys=dict(
            # Backbone learns even more slowly than head — preserves the
            # visual features learned from synthetic pretraining.
            backbone=dict(lr_mult=0.5),
        )))

train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=10)

# Geometric-loss warmup: model heatmaps are already well-formed from
# stage 1, so we can activate the geometric prior quickly instead of the
# usual 100-epoch warmup. 200 steps ≈ 11 epochs at 288/batch=16.
model = dict(
    head=dict(
        loss=dict(warmup_steps=200)))

# Shorter warmup + shorter cosine to match the shorter schedule.
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False,
         begin=0, end=50),
    dict(type='CosineAnnealingLR', eta_min=0, begin=0, end=150,
         by_epoch=True),
]
