"""STAGE 2 for the v2 pretrain: finetune on real, from the wide-synthetic init.

The ONLY change from hrnet-w32_8-kp_udp_rgbd_geom_finetune_chewy.py is which
stage-1 checkpoint it starts from. Data, augmentation, optimizer, schedule,
domain balance, evaluators and save_best are all inherited unchanged, so any
difference in the result is attributable to the pretrain corpus and nothing
else. That is the whole point of splitting this into its own config rather
than editing `load_from` in place.

BEFORE LAUNCHING: set `load_from` below to the checkpoint stage 1 actually
selected. It will NOT be epoch 70 just because the last pretrain's was --
read the run's best_coco_AP checkpoint name off MLflow or work_dirs. If the
path is wrong mmengine raises rather than silently starting from ImageNet,
so a typo fails loudly, but a path pointing at the OLD pretrain would not.

HOW TO LAUNCH
    python tools/train.py \
      configs/hrnet-w32_8-kp_udp_rgbd_geom_finetune_chewy_synthv2.py \
      --work-dir work_dirs/hrnet_chewy2_synthv2

    Deliberately NO max_keep_ckpts here. Sweeping the periodic checkpoints
    after the run is how the sigma experiment showed `max/cyclic_mean_px`
    peaked at epoch 20 and then degraded 63% to the final epoch -- capping to
    the last 3 would have hidden that and shipped a much worse model. At
    interval 10 over 150 epochs it is 15 files x 329MB, about 5GB, which is
    cheap against re-running to recover a checkpoint you deleted.

WHAT TO COMPARE AGAINST  (the current shipping baseline, from the same
finetune recipe on the old pretrain):

    chewy pair_sep_slope   0.696       <- the metric the wide set targets
    rsc   pair_sep_slope   0.076       <- worst domain, most headroom
    chewy median_px        2.667
    chewy n_over_20px         17
    max/cyclic_mean_px     6.738 @ep50 <- ship-selection metric

Read `pair_sep_slope` first. If it has not moved off 0.70 / 0.08, the
synthetic prior was not the mechanism behind corner collapse, and no amount
of further widening will help -- stop and go back to the cross-view selector.
"""
_base_ = ['./hrnet-w32_8-kp_udp_rgbd_geom_finetune_chewy.py']

# TODO set to the stage-1 v2 checkpoint before launching.
load_from = 'work_dirs/hrnet_synth_pretrain_v2/best_coco_AP_epoch_70.pth'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SafeMLflowVisBackend',
        exp_name='box-flap-pose',
        run_name='rgbd-geom-finetune-chewy2-from-synthv2',
        tracking_uri='http://10.200.104.2:31016',
        tags=dict(
            model='hrnet-w32',
            variant='rgbd-geom',
            data='chewy2+rsc',
            stage='finetune',
            pretrain='synth-v2-wide-union'),
        artifact_suffix=('.py', '.pth', '.json')),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
