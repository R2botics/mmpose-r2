_base_ = ['./hrnet-w32_8-kp_udp_rgbd.py']

# Register the new geometric loss alongside the existing custom imports
custom_imports = dict(
    imports=[
        'mmpose.datasets.transforms.rgbd',
        'mmpose.models.data_preprocessors.nchannel',
        'mmpose.models.losses.geometric_loss',
    ],
    allow_failed_imports=False)

# Only change vs the RGBD config: swap the head's loss for the geometric one.
# Soft-perp / soft-parallel each penalised only beyond a 10° tolerance zone,
# leaving room for the slight perspective in our top-down cameras.
model = dict(
    head=dict(
        loss=dict(
            _delete_=True,
            type='GeometricKeypointMSELoss',
            use_target_weight=True,
            # 6x weaker than the first attempt — geometric loss should be a
            # polish signal, not a primary objective.
            perp_weight=0.005,
            parallel_weight=0.005,
            tolerance_deg=10.0,
            softmax_temp=10.0,
            # 18 train batches/epoch * 100 epochs ≈ 1800 steps of warmup.
            # Lets the model fully converge on heatmap MSE before any geometric
            # constraint kicks in. Peak AP for RGBD-only landed at epoch 20, so
            # 100 epochs of warmup gives ample headroom.
            warmup_steps=1800)))
