_base_ = ['./hrnet-w32_8-kp.py']

# UDPHeatmap: sub-pixel precision via UDP framework, fast encoding
codec = dict(
    type='UDPHeatmap',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2)

model = dict(head=dict(decoder=codec))

train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(
                type='RandomBBoxTransform',
                shift_factor=0.1,
                scale_factor=[0.75, 1.25],
                rotate_factor=30),
            dict(type='TopdownAffine', input_size=(256, 256), use_udp=True),
            dict(
                type='PhotometricDistortion',
                brightness_delta=32,
                contrast_range=(0.6, 1.4),
                saturation_range=(0.6, 1.4),
                hue_delta=18),
            dict(type='GenerateTarget', encoder=codec),
            dict(type='PackPoseInputs')
        ]))
