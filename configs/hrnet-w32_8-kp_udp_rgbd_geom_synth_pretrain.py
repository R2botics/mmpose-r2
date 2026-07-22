_base_ = ['./hrnet-w32_8-kp_udp_rgbd_geom.py']

# STAGE 1 of two-stage training: pretrain on synthetic data only.
# Model starts from the 5-channel ImageNet-inflated backbone and learns
# to detect box corners on the diverse synthetic distribution. The
# resulting checkpoint feeds into the stage-2 config which fine-tunes on
# real data.
#
# Watch: val AP here will be lower than real-only training because
# validation happens on the REAL val set — this measures how well
# synthetic-learned features transfer to real, which is exactly the
# question we want stage 2 to answer.

# MLflow — separate run for the two-stage story
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
        run_name='rgbd-geom-synth-pretrain',
        tracking_uri='http://10.200.104.2:31016',
        tags=dict(
            model='hrnet-w32',
            variant='rgbd-geom',
            data='synthetic-only',
            train_size='168',
            stage='pretrain',
            backbone_init='imagenet'),
        artifact_suffix=['.py', '.pth', '.json']),
]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# Train on synthetic only. The synthetic images are symlinked under
# data/RSC_Keypoints_RGBD/images/synthetic/ and depth/synthetic/, and the
# annotation JSON references those paths.
train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/person_keypoints_synth_only.json'))

# Fewer epochs — synthetic is smaller and more homogeneous than real+synth.
# 200 epochs at 168/batch=16 = ~2100 iterations. Long enough to converge
# but short enough that we don't over-specialize on synthetic.
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=10)

# Adjust the geometric-loss warmup to a shorter schedule.
# At 168 images / batch 16 = ~11 iters/epoch, so 1100 steps ≈ 100 epochs.
model = dict(
    head=dict(
        loss=dict(warmup_steps=1100)))

# Scale the cosine schedule to the new epoch count.
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False,
         begin=0, end=100),
    dict(type='CosineAnnealingLR', eta_min=0, begin=0, end=200,
         by_epoch=True),
]
