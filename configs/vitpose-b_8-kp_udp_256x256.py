"""ViTPose-B, 8 box-flap keypoints, 256x256, N-channel input.

The direct counterpart of `hrnet-w32_8-kp_udp_rgbd_geom_finetune_chewy.py`:
same codec, same sigma, same input size, same geometric loss, same per-domain
minimax selection. ONLY the backbone differs, so a difference in the metrics is
attributable to the architecture rather than to a re-tuned recipe.

This file is the 5-channel (RGB + height + validity) production variant. The
depth ablation lives in three thin children that change one number each:

    vitpose-b_8-kp_udp_rgb.py     3ch  [R,G,B]
    vitpose-b_8-kp_udp_rgbd.py    4ch  [R,G,B,height]
    vitpose-b_8-kp_udp_rgbdm.py   5ch  [R,G,B,height,valid]   (== this file)


WHAT TO EXPECT FROM THIS ARCHITECTURE -- read before spending GPU-hours
======================================================================
Corner-pair collapse is unlikely to improve here, and the reason is a number
you can compute from the annotations without training anything. Measured
through this config's OWN val pipeline, i.e. in the 256x256 tensor the network
actually receives -- separation between the two keypoints marking one physical
box corner:

                        RSC train    RSC val      OOD
    n pairs                   972        241      120
    median, input px         6.25       6.48     6.82
    p90,    input px        19.51      11.73    11.93

    the same distance in the units each stage works in
    heatmap px (stride 4)    1.56       1.62     1.71     <- sigma is 2
      % under 1 sigma        64.3       66.0     65.8
      % under 2 sigma        88.4       93.8     98.3
    ViT tokens (stride 16)   0.39       0.40     0.43
      % under 1 patch width  88.4       93.8     98.3

Read the last two rows, and note they are the SAME numbers. That is not a
coincidence: 2 sigma of the heatmap target is 4 heatmap px = 16 input px = one
patch. The blob the loss is written in and the patch the ViT tokenizes with are
the same physical scale. So 88-98% of corner pairs are separated by less than
one patch width -- most of them share a patch outright, and the rest straddle a
single boundary. A plain ViT's first operation is a linear
projection of each patch to a single 768-d token; from that point on there is
no spatial detail below stride 16 anywhere in the backbone, and the decoder is
asked to reconstruct a sub-token separation that the tokenizer discarded.
HRNet-W32 keeps a stride-4 branch end to end, so the same separation spans ~1.6
feature pixels rather than ~0.4 -- still small, but the information is at least
still present in the feature map. On this specific failure the plain ViT is
structurally the worse instrument, and it is worth being clear about that
before the run rather than after.

Two honest qualifications, in both directions:

  * The BINDING constraint is not the backbone. Both architectures decode a
    64x64 heatmap with sigma=2, so both spend their supervision on blobs whose
    +/-2-sigma support (8 heatmap px) is four times the median separation they
    are supposed to resolve. `..._sigma15.py` already measured this from the other
    direction, and the table above puts 64-66% of pairs closer than one sigma. Changing the backbone while
    leaving sigma at 2 changes the instrument, not the limit. Expect a
    backbone swap to move `mean_px` and to leave `frac_collapsed_pairs` and
    `corner_span_err_px` roughly where they are; if it does, that is the
    experiment succeeding at telling you something, not failing.

  * There IS something a ViT should be better at here, and it is not
    collapse. Global attention from layer 1 is a good fit for deciding WHICH
    flap is which -- the major/minor assignment that `n_clock90` counts, which
    is a whole-box judgement (aspect ratio, which flaps are raised,
    foreshortening) that HRNet's local convolutions have to assemble
    hierarchically. If `n_clock90` and the gap between `mean_px` and
    `cyclic_mean_px` shrink, that is ViTPose earning its place, and it is
    worth watching independently of the collapse numbers.

If the sub-token argument is what you want to attack directly, the lever is
`ratio=2` in the backbone: it keeps the 16x16 kernel but halves the stride, so
a 256x256 input yields a 32x32 token grid at stride 8. The median pair becomes
~0.8 tokens instead of ~0.4, and the share falling inside a single patch drops
from 88-98% to 64-66% -- an improvement, but note it does NOT get the pair into
separate patches in the median case. Even stride 8 is coarse against this
distribution. It costs roughly 16x the attention FLOPs
(1024 tokens vs 256) and no pretrained pos_embed matches it without the
resampling in `scripts/convert_vit_pretrain.py`. That is the honest ViT-native
answer to "the backbone is too coarse" -- and it is one line below.


CHECKPOINT
==========
`load_from` / `init_cfg` expect a checkpoint produced by
`scripts/convert_vit_pretrain.py`, which renames `blocks.`->`layers.` and
resamples `pos_embed` onto the 16x16 grid. It does NOT widen the patch embed
to 5 channels: the extra channels go through `ViT.extra_stem`, a separate
zero-initialised projection, so ONE checkpoint serves all three ablation arms
and the 3ch/4ch/5ch runs share an identical RGB tokenizer. The reasoning is in
`mmpose/models/backbones/vit.py`.
"""
# Local path, not `mmpose::_base_/...`. Running `python tools/train.py` puts
# tools/ at sys.path[0], so `mmpose` resolves to whichever copy is in
# site-packages -- and that one has no `.mim/model-index.yml`, which is what
# `mmpose::` needs to resolve. The file here is byte-identical to the packaged
# one; this just removes a dependency on how the interpreter was launched.
_base_ = ['./_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'mmpose.models.backbones.vit',
        'mmpose.datasets.transforms.rgbd',
        'mmpose.models.data_preprocessors.nchannel',
        'mmpose.models.losses.geometric_loss',
        'mmpose.evaluation.metrics.multi_domain_coco_metric',
        'mmpose.evaluation.metrics.multi_domain_distance_metric',
    ],
    allow_failed_imports=False)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# --------------------------------------------------------------------- codec
# Deliberately IDENTICAL to the HRNet baseline, sigma included. sigma is the
# lever most likely to move corner collapse (see the docstring), which is
# exactly why it is not being changed in the same experiment as the backbone.
# Sweep it on whichever backbone wins, not during the comparison.
codec = dict(
    type='UDPHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

pretrained_ckpt = 'work_dirs/pretrained/vit_base_256x256.pth'

# ---------------------------------------------------------------- input spec
# Overridden wholesale by the three ablation children.
in_channels = 5
#                R        G       B      height  valid
img_mean = [123.675, 116.28, 103.53, 414.2, 127.5]
img_std = [58.395, 57.12, 57.375, 184.9, 127.5]

# HEADS UP -- the depth constants above are inherited verbatim from
# `hrnet-w32_8-kp_udp_rgbd.py` and they do not match this data. Measured over
# the depth PNGs (RSC: 120 files, OOD: all 30):
#
#                       all px          >0 px          zero
#       RSC    mean 37.6  std  84.9    mean 193 std  84    80.5%
#       OOD    mean 61.9  std 125.3    mean 279 std 101    77.8%
#
# With mean=414.2 / std=184.9 the ENTIRE RSC depth channel lands in
# z in [-2.24, -0.28]: it never crosses zero, sits ~2 sigma off centre, and
# uses about 1.5 sigma of range where RGB uses ~4. The constant offset is
# largely absorbed by the patch-embed bias and the first LayerNorm, so this is
# not fatal -- but the compressed scale does shrink depth's gradient
# contribution relative to RGB, which biases a depth ablation AGAINST depth.
#
# It is left UNCHANGED here on purpose: every HRNet number this project has
# was produced with these constants, and re-scaling depth in the same
# experiment that swaps the backbone would confound both. Measured
# alternatives, for a separate single-variable run once the backbone question
# is settled:
#
#       img_mean = [123.675, 116.28, 103.53,  37.6, 127.5]
#       img_std  = [ 58.395,  57.12,  57.375,  84.9, 127.5]
#
# Note what those do to the zeros. 80% of depth pixels are exactly 0 and would
# map to z = -0.44 instead of -2.24 -- still a single distinctive constant, so
# the "a zero is not a measurement" signal survives either way. That signal is
# the reason not to hole-fill: an interpolated value is indistinguishable from
# a measured one, and channel 4 is what makes the distinction legible.

# --------------------------------------------------------------------- model
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='NChannelPoseDataPreprocessor',
        mean=img_mean,
        std=img_std),
    backbone=dict(
        type='ViT',
        img_size=(256, 256),
        patch_size=16,
        in_channels=in_channels,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        # ratio=2 -> stride-8 tokens (32x32). See the docstring: this is the
        # one knob that makes a plain ViT operate below the patch scale, at
        # ~16x the attention cost. Requires a pos_embed converted with
        # `--ratio 2`.
        ratio=1,
        drop_path_rate=0.3,
        # Zero-initialised parallel stem for channels 3.. At step 0 the model
        # is bit-identical to the pretrained RGB one; 'inflate_mean'
        # reproduces the HRNet-style seeding instead.
        extra_stem_mode='zero',
        use_checkpoint=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_ckpt)),
    head=dict(
        type='HeatmapHead',
        in_channels=768,
        out_channels=8,
        # ViTPose's "classic" decoder: two 4x4 deconvs, 16x16 -> 64x64, so the
        # heatmap stride matches HRNet's exactly and the codec floor is the
        # same for both. The 3x3 final conv is ViTPose's `final_conv_kernel=3`.
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        final_layer=dict(kernel_size=3, padding=1),
        loss=dict(
            type='GeometricKeypointMSELoss',
            use_target_weight=True,
            perp_weight=0.005,
            parallel_weight=0.005,
            tolerance_deg=10.0,
            softmax_temp=10.0,
            # 288/16 = 18 iters per epoch, so 200 steps ~= 11 epochs of pure
            # heatmap MSE before the soft-argmax terms engage.
            warmup_steps=200,
            # Measured on the HRNet runs: without this, 169 corners that are
            # NOT in the image still peak at a mean of 0.83. Carried over
            # because it is the production recipe, not because it interacts
            # with the backbone.
            absent_weight=1.0,
            # OFF, matching the HRNet baseline. It optimises soft-argmax
            # coordinates while inference decodes with argmax+DARK, and it was
            # measured not to move the along-edge error (4.08 -> 4.07 px).
            span_weight=0.0),
        decoder=codec),
    test_cfg=dict(flip_test=False))

# ------------------------------------------------------------------ datasets
dataset_type = 'CocoDataset'
data_mode = 'topdown'
metainfo_file = 'configs/_base_/datasets/RSC_Keypoints.py'

rsc_root = 'data/RSC_Keypoints_RGBD'
ood_root = 'data/OOD'
# Mount the customer sets and uncomment the four marked blocks below to add
# them. They are the SECOND selectable domain, and without one the minimax in
# the evaluator degenerates to a single domain -- it still reports correctly,
# it just has nothing to be worst of.
# chewy_root = 'data/ChewyCrops2'     # train, captured 2026-08-18
# chewy_val_root = 'data/ChewyCrops'  # val, held-out SESSION 2026-08-13

train_pipeline = [
    dict(type='LoadRGBDImage', channels=in_channels),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomBBoxTransform',
         shift_factor=0.1, scale_factor=[0.75, 1.25], rotate_factor=30),
    dict(type='TopdownAffine', input_size=(256, 256), use_udp=True),
    dict(type='PhotometricDistortionRGBOnly',
         brightness_delta=32, contrast_range=(0.6, 1.4),
         saturation_range=(0.6, 1.4), hue_delta=18),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadRGBDImage', channels=in_channels),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256), use_udp=True),
    dict(type='PackPoseInputs'),
]


def _subset(root, ann, test_mode=False):
    return dict(
        type=dataset_type, data_root=root, ann_file=ann,
        data_prefix=dict(img='images/'),
        metainfo=dict(from_file=metainfo_file),
        test_mode=test_mode,
        pipeline=[])          # CombinedDataset owns the shared pipeline


train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file=metainfo_file),
        datasets=[
            _subset(rsc_root, 'annotations/person_keypoints_train.json'),
            # _subset(chewy_root, 'annotations/person_keypoints_Train.json'),
        ],
        sample_ratio_factor=[1.0],
        # sample_ratio_factor=[1.0, 1.0],   # with chewy
        pipeline=train_pipeline))

# ---------------------------------------------------------------- validation
# Per domain, never pooled. OOD is reported but carries `select=False` so it
# can never drive `save_best` -- it is the held-out set, and a checkpoint
# chosen on it stops being an honest estimate of anything.
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file=metainfo_file),
        datasets=[
            _subset(rsc_root, 'annotations/person_keypoints_val.json', True),
            _subset(ood_root, 'annotations/person_keypoints_Test.json', True),
            # _subset(chewy_val_root, 'annotations/person_keypoints_Train.json', True),
        ],
        pipeline=val_pipeline))

val_evaluator = [
    dict(
        type='MultiDomainKeypointDistanceMetric',
        domains=[
            dict(name='rsc', data_root=rsc_root),
            # OOD: reported, never selected on.
            dict(name='ood', data_root=ood_root, select=False),
            # dict(name='chewy', data_root=chewy_val_root),
        ],
        fail_thr_px=20.0),
    # AP, for continuity with the historical runs only. It is OKS-based and
    # structurally blind to corner collapse -- a collapsed pair reads as two
    # mediocre keypoints -- and on this project's full-image bboxes it
    # saturates near 1.0. Never select on it.
    dict(
        type='MultiDomainCocoMetric',
        domains=[
            dict(name='rsc', data_root=rsc_root,
                 ann_file=f'{rsc_root}/annotations/person_keypoints_val.json'),
            dict(name='ood', data_root=ood_root,
                 ann_file=f'{ood_root}/annotations/person_keypoints_Test.json'),
        ]),
]

test_cfg = dict()
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# ---------------------------------------------------------------- selection
# Two best-checkpoints are kept, both minimax over the SELECTABLE domains and
# both 'less':
#
#   best_max_cyclic_mean_px_*            keypoint accuracy, the HRNet
#                                        baseline's criterion -- kept so the
#                                        two architectures are ranked by the
#                                        same rule.
#   best_max_cyclic_corner_span_err_px_* median |pred_span - gt_span| on the
#                                        corner pairs. This is the product
#                                        measurement, scored directly.
#
# They are kept separately rather than combined because every proxy in this
# project has at some point disagreed with the others, and a blended score
# hides which one moved. Ship on the span one; use the pixel one to explain it.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best=['max/cyclic_corner_span_err_px', 'max/cyclic_mean_px'],
        rule='less'))

# ---------------------------------------------------------------- schedule
# Layer-wise LR decay, the ViTPose recipe. Two things to know about it:
#
#  * `LayerDecayOptimWrapperConstructor` identifies blocks by the literal
#    prefix `backbone.layers.` -- which is why this repo's ViT names them
#    `layers` and not ViTPose's `blocks`. With `blocks` every transformer
#    layer falls into the constructor's else-branch at lr_scale 1.0 and the
#    decay silently does nothing.
#  * It maps `backbone.patch_embed*` to layer 0, i.e. 0.75**13 = 0.024x base
#    LR. `backbone.extra_stem` is deliberately NOT under that prefix, so the
#    zero-initialised depth stem trains at full LR. Under layer 0 it would
#    barely leave zero and the depth ablation would report "depth does not
#    help" when it had measured "depth never trained".
#
# lr 1e-4, not ViTPose's COCO 5e-4: that number is for batch 512 over 150k
# images. This is batch 16 over 288.
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1),
    constructor='LayerDecayOptimWrapperConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.75),
    clip_grad=dict(max_norm=1.0, norm_type=2))

# 150 epochs matches the HRNet finetune schedule so the comparison stays clean.
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=10)
val_cfg = dict()

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=50),
    dict(type='CosineAnnealingLR', eta_min=0, begin=0, end=150, by_epoch=True),
]
