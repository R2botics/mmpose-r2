"""STAGE 1 for ViTPose-B: pretrain the 5-channel arm on synthetic.

WHY THIS EXISTS
    The ablation arms (vitpose-b_8-kp_udp_{rgb,rgbd,rgbdm}.py) train on RSC
    real ONLY -- 288 images against an 86M-parameter ViT. That will memorise
    the set. HRNet needed the synthetic pretrain to work at all, and a ViT of
    this size needs it more, not less.

    Same corpus and same split as the HRNet v2 pretrain, so the two backbones
    are comparable end to end:
        vms_flaps_cropped (wide)  11,032 after holdout
        old synth (narrow)        18,130 after holdout
        ------------------------------
        29,162 -> 911 iters/epoch at batch 32

WHAT THIS FIXES vs THE ARM IT INHERITS FROM
    The arm's LR warmup is `LinearLR ... end=50` -- fifty ITERATIONS, which is
    ~3 epochs at 18 iters/epoch on 288 images. At 911 iters/epoch it is 5% of
    one epoch, and a ViT hit with full LR that early diverges or lands in a bad
    basin. Warmup here is 2000 iters (~2.2 epochs). This is the single most
    likely cause of a failed ViT pretrain, and it is silent -- the loss just
    never comes down.

    `warmup_steps` on the geometric loss is likewise specified in ITERATIONS
    and must track dataset size to keep meaning "ramp over the first half".

VALIDATION
    Judged on generalisation, not on RSC val alone -- stage 1 trains on zero
    real images, so all three real sets are legitimately held out:
        rsc 72 / chewy 76 / ood 30   = 178 imgs, 428 corner pairs
    Plus a held-out slice of EACH synthetic set, carrying `select=False` so it
    is reported but can never drive save_best. If synthetic val keeps improving
    while the real sets degrade, that is the generator being memorised and the
    signal to stop -- exactly what happened to the HRNet v2 pretrain after
    epoch 30.

    Selection is on `max/cyclic_corner_span_err_px` (inherited): the product
    measurement -- median |pred_span - gt_span| -- scored directly, minimax
    over the selectable domains. Note per-keypoint error and span error have
    already disagreed once on this project, and span error is the deliverable.

BEFORE LAUNCHING
    1. The converted ViT weights must exist -- the arm's init_cfg points at
       work_dirs/pretrained/vit_base_256x256.pth:
           wget -P work_dirs/pretrained/ \\
             https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
           python scripts/convert_vit_pretrain.py \\
             --src work_dirs/pretrained/mae_pretrain_vit_base.pth \\
             --dst work_dirs/pretrained/vit_base_256x256.pth --input-size 256 256
    2. The synthetic splits must exist:
           python scripts/make_synth_val_split.py --n 400 \\
             ~/Documents/vms_flaps_cropped/annotations/person_keypoints_train.json \\
             data/RSC_Keypoints_RGBD/annotations/person_keypoints_synth_only.json
    3. python scripts/verify_val_setup.py configs/vitpose-b_8-kp_udp_rgbdm_synth_pretrain.py

HOW TO LAUNCH
    python tools/train.py \\
      configs/vitpose-b_8-kp_udp_rgbdm_synth_pretrain.py \\
      --work-dir work_dirs/vitpose_synth_pretrain

    BATCH SIZE is 32 to match the HRNet pretrain. ViT-B at 256x256 needs
    roughly 2-3x HRNet-W32's activation memory (HRNet ran at 6.6GB), so if it
    OOMs, drop to 16 -- but then iters/epoch doubles to 1823 and BOTH
    `warmup_steps` (-> 91150) and the LinearLR `end` (-> 4000) must double with
    it, or the schedule silently means something different.
    `use_checkpoint=True` on the backbone is the cheaper fix: it trades ~30%
    step time for a large activation-memory saving and leaves the schedule
    alone.
"""
_base_ = ['./vitpose-b_8-kp_udp_rgbdm.py']

# `imports` is a LIST, so declaring custom_imports here REPLACES the arm's
# rather than extending it -- every entry has to be repeated. The addition is
# safe_mlflow: the ViT arms never logged to MLflow, so the backend this config
# adds below was unregistered and the run died in build_visualizer.
custom_imports = dict(
    imports=[
        'mmpose.models.backbones.vit',
        'mmpose.datasets.transforms.rgbd',
        'mmpose.models.data_preprocessors.nchannel',
        'mmpose.models.losses.geometric_loss',
        'mmpose.evaluation.metrics.multi_domain_coco_metric',
        'mmpose.evaluation.metrics.multi_domain_distance_metric',
        'mmpose.engine.vis_backends.safe_mlflow',
    ],
    allow_failed_imports=False)

# ------------------------------------------------------------------ the data
new_synth_root = '/home/rsquared/Documents/vms_flaps_cropped'
new_synth_ann = 'annotations/person_keypoints_train_trainsplit.json'
new_synth_val = 'annotations/person_keypoints_train_valsplit.json'

old_synth_root = 'data/RSC_Keypoints_RGBD'
old_synth_ann = 'annotations/person_keypoints_synth_only_trainsplit.json'
old_synth_val = 'annotations/person_keypoints_synth_only_valsplit.json'

rsc_root = 'data/RSC_Keypoints_RGBD'
chewy_root = 'data/ChewyCrops'
ood_root = 'data/OOD'

metainfo_file = 'configs/_base_/datasets/RSC_Keypoints.py'

# POST-SPLIT counts (400 images held out of each). Used only to size the
# schedules below; an out-of-date number changes them silently.
N_NEW = 11432 - 400
N_OLD = 18530 - 400
BATCH_SIZE = 32
_n_samples = N_NEW + N_OLD
_iters_per_epoch = _n_samples // BATCH_SIZE          # 911

# Old synthetic images live UNDER the real RSC tree, so `synth_old` is routed
# on the deeper path and `rsc` keeps the shallow root. Routing is longest
# prefix first; only the METRIC's key is the deeper path, not the dataset's.
old_synth_route = f'{old_synth_root}/images/synthetic'


def _subset(root, ann, test_mode=False):
    d = dict(
        type='CocoDataset',
        data_root=root,
        ann_file=ann,
        data_prefix=dict(img='images/'),
        metainfo=dict(from_file=metainfo_file),
        pipeline=[])          # CombinedDataset owns the pipeline
    if test_mode:
        d['test_mode'] = True
    return d


train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=8,
    dataset=dict(
        _delete_=True,
        type='CombinedDataset',
        metainfo=dict(from_file=metainfo_file),
        datasets=[
            _subset(new_synth_root, new_synth_ann),
            _subset(old_synth_root, old_synth_ann),
        ],
        # [new, old] in natural proportion -- the wide set is 38% of a batch.
        sample_ratio_factor=[1.0, 1.0],
        pipeline={{_base_.train_pipeline}}))

_eval_subsets = [
    _subset(rsc_root, 'annotations/person_keypoints_val.json', test_mode=True),
    _subset(chewy_root, 'annotations/person_keypoints_Train.json', test_mode=True),
    _subset(ood_root, 'annotations/person_keypoints_Test.json', test_mode=True),
    _subset(new_synth_root, new_synth_val, test_mode=True),
    _subset(old_synth_root, old_synth_val, test_mode=True),
]

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='CombinedDataset',
        metainfo=dict(from_file=metainfo_file),
        datasets=_eval_subsets,
        pipeline={{_base_.val_pipeline}}))

_val_domains = [
    dict(name='rsc', data_root=rsc_root),
    dict(name='chewy', data_root=chewy_root),
    dict(name='ood', data_root=ood_root),
    dict(name='synth_new', data_root=new_synth_root, select=False),
    dict(name='synth_old', data_root=old_synth_route, select=False),
]
_val_domains_coco = [
    dict(name='rsc', data_root=rsc_root,
         ann_file=f'{rsc_root}/annotations/person_keypoints_val.json'),
    dict(name='chewy', data_root=chewy_root,
         ann_file=f'{chewy_root}/annotations/person_keypoints_Train.json'),
    dict(name='ood', data_root=ood_root,
         ann_file=f'{ood_root}/annotations/person_keypoints_Test.json'),
    dict(name='synth_new', data_root=new_synth_root, select=False,
         ann_file=f'{new_synth_root}/{new_synth_val}'),
    dict(name='synth_old', data_root=old_synth_route, select=False,
         ann_file=f'{old_synth_root}/{old_synth_val}'),
]

val_evaluator = [
    dict(type='MultiDomainKeypointDistanceMetric', domains=_val_domains),
    dict(type='MultiDomainCocoMetric', domains=_val_domains_coco),
]

# `_delete_` belongs at the TOP level only: mmengine strips it while merging a
# child into a base dict, and the base sets test_dataloader to None, so a
# nested one survives into CombinedDataset.__init__ and raises. Nothing is
# inherited either, so the sampler and loader flags are spelled out.
test_cfg = dict(_delete_=True)
test_dataloader = dict(
    _delete_=True,
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file=metainfo_file),
        datasets=_eval_subsets,
        pipeline={{_base_.val_pipeline}}))
test_evaluator = val_evaluator

# --------------------------------------------------------------- schedules
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=10)

# Geometric loss ramps 0 -> 1 over the first HALF of training. The arm's 200
# was sized for 288 images; recomputed here or it would finish by epoch 11.
model = dict(head=dict(loss=dict(warmup_steps=_iters_per_epoch * 50)))

# LinearLR end=2000 iters (~2.2 epochs), NOT the arm's 50. See the docstring:
# a ViT taken to full LR inside 5% of one epoch is the classic silent failure.
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False,
         begin=0, end=2000),
    dict(type='CosineAnnealingLR', eta_min=0, begin=0, end=100, by_epoch=True),
]

# ------------------------------------------------------------------ tracking
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SafeMLflowVisBackend',
        exp_name='box-flap-pose',
        run_name='vitpose-b-rgbdm-synth-pretrain',
        tracking_uri='http://10.200.104.2:31016',
        tags=dict(
            model='vitpose-b',
            variant='rgbdm',
            data='synth-union-wide+old',
            train_size=str(_n_samples),
            stage='pretrain',
            backbone_init='mae'),
        artifact_suffix=('.py', '.pth', '.json')),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
