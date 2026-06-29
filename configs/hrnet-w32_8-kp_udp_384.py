_base_ = ['./hrnet-w32_8-kp.py']

# UDP at 384x384 input -> 96x96 heatmap (50% finer than baseline 64x64)
codec = dict(
    type='UDPHeatmap',
    input_size=(384, 384),
    heatmap_size=(96, 96),
    sigma=2)

model = dict(head=dict(decoder=codec))

# 384x384 needs ~2.25x more memory than 256x256, so cut batch in half
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(
                type='RandomBBoxTransform',
                shift_factor=0.1,
                scale_factor=[0.75, 1.25],
                rotate_factor=30),
            dict(type='TopdownAffine', input_size=(384, 384), use_udp=True),
            dict(
                type='PhotometricDistortion',
                brightness_delta=32,
                contrast_range=(0.6, 1.4),
                saturation_range=(0.6, 1.4),
                hue_delta=18),
            dict(type='GenerateTarget', encoder=codec),
            dict(type='PackPoseInputs')
        ]))

val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(384, 384), use_udp=True),
            dict(type='PackPoseInputs')
        ]))
