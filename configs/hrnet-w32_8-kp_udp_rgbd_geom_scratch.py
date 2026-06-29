_base_ = ['./hrnet-w32_8-kp_udp_rgbd_geom.py']

# Train the full RGBD + geometric-loss model from scratch — no pretrained
# weights of any kind. The inflated 5-channel checkpoint we built from
# HRNet+ImageNet is also replaced with random Kaiming initialization, so
# nothing in this run derives from ImageNet.
#
# Useful as a baseline to isolate how much our final accuracy depends on
# the ImageNet pretraining we'd otherwise rely on.
model = dict(
    backbone=dict(
        # `_delete_=True` removes the inherited init_cfg dict so the
        # mmengine merge doesn't fall back to the parent's Pretrained path.
        init_cfg=dict(_delete_=True, type='Kaiming', layer='Conv2d')),
    head=dict(
        loss=dict(
            # Push the geometric-loss warmup further out. The previous
            # warmup_steps=1800 (~100 epochs) assumed the model had already
            # converged on heatmap MSE by then. From scratch, the heatmaps
            # take much longer to develop peaks, so we delay the geometric
            # prior until ~epoch 300.
            warmup_steps=5400)))

# From-scratch training needs significantly more epochs. The pretrained run
# peaked at epoch ~30; without pretraining, expect peak somewhere between
# epoch 200 and 500 (if it converges meaningfully at all).
train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=10)

# Gentler warmup — random init is much more fragile than pretrained.
# Default lr=5e-4 from a Kaiming-init backbone can diverge in the first few
# iterations because the first conv produces near-zero activations until the
# scale settles, and gradients can be huge.
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-6,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        begin=0,
        end=1000,
        by_epoch=True),
]
