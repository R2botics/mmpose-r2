_base_ = ['./hrnet-w32_8-kp_udp_rgbd_geom.py']

# Register the sanitizing MLflow backend
custom_imports = dict(
    imports=[
        'mmpose.datasets.transforms.rgbd',
        'mmpose.models.data_preprocessors.nchannel',
        'mmpose.models.losses.geometric_loss',
        'mmpose.engine.vis_backends.safe_mlflow',
    ],
    allow_failed_imports=False)

# Train on real + synthetic combined (288 real + 556 synthetic = 844 total).
# Everything else (loss, backbone, augmentation, pretrained init) is inherited
# from the RGBD+Geom config that scored 0.9752 on lab data.
#
# Synthetic images are symlinked under images/synthetic/ and depth/synthetic/,
# so LoadRGBDImage picks them up via the same directory-swap convention.
train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/person_keypoints_train_combined.json'))

# Validation stays on the real val set only — we want to know how the model
# does on real production-style images, not on synthetic ones.

# ---------- MLflow tracking ----------
# Adds MLflow alongside TensorBoard/local logging. All scalars logged by
# mmengine (train loss, val coco/AP, coco/AP@0.75, learning rate, etc.) get
# streamed to the MLflow tracking server, plus the config file as an artifact.
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SafeMLflowVisBackend',
        exp_name='box-flap-pose',              # groups related runs
        run_name='rgbd-geom-combined-v2-new-synth',   # this specific run
        tracking_uri='http://10.200.104.2:31016',
        tags=dict(
            model='hrnet-w32',
            variant='rgbd-geom',
            data='real+synthetic',
            train_size='844',
            backbone_init='imagenet'),
        artifact_suffix=['.py', '.pth', '.json']),
]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
