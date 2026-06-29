_base_ = ['./hrnet-w32_8-kp.py']

# Only change vs base: enable DARK (Distribution-Aware coordinate Representation)
# This adds sub-pixel refinement to the heatmap-to-keypoint decoding step.
# No memory cost, no extra training time — purely a smarter decode.
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2,
    unbiased=True)

# Re-link the codec into the head decoder and the train/val pipeline targets
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
