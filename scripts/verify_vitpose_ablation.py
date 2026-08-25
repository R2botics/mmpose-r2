#!/usr/bin/env python3
"""Preflight for the ViTPose depth ablation. Run it BEFORE spending GPU-hours.

The three configs

    configs/vitpose-b_8-kp_udp_rgb.py     3ch
    configs/vitpose-b_8-kp_udp_rgbd.py    4ch
    configs/vitpose-b_8-kp_udp_rgbdm.py   5ch

only mean something as an ablation if they are identical apart from the channel
count. They are separate files with hand-copied pipelines, so that is an
assumption, and assumptions of this shape fail quietly: a stale augmentation
parameter in one arm does not raise, it just moves the number you were about to
attribute to depth.

This script checks the four things that can silently invalidate the comparison:

  1. CONFIG DIFF -- the resolved configs differ only in fields whitelisted as
     channel-related. Anything else is a hard failure.
  2. WIRING -- `LoadRGBDImage.channels`, `data_preprocessor.mean/std` length and
     `backbone.in_channels` agree, in every pipeline of every arm. A mismatch
     here trains on silently misnormalised channels.
  3. STEP-0 EQUIVALENCE -- with the same pretrain loaded, the 5ch model's
     tokens are BIT-identical to the 3ch model's on the same RGB input. This is
     what `extra_stem_mode='zero'` buys, and it is the reason the three arms
     are comparable at all.
  4. LEARNING RATE -- `extra_stem` is not swallowed by the layer-0 group of
     `LayerDecayOptimWrapperConstructor`. At layer 0 it would train at ~2.4% of
     base LR, a zero-initialised stem would never leave zero, and the ablation
     would report "depth does not help" having measured "depth never trained".

Usage:
    python scripts/verify_vitpose_ablation.py
    python scripts/verify_vitpose_ablation.py --skip-pretrained   # no ckpt yet
"""
import argparse
import sys
from pathlib import Path

# Running `python scripts/foo.py` puts scripts/ -- not the repo root -- at
# sys.path[0], so a `mmpose` installed in site-packages wins over this repo's.
# That one differs: it has no `.mim/model-index.yml`, so the `mmpose::` base in
# the configs fails to resolve. Put the repo root first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

CONFIGS = {
    3: 'configs/vitpose-b_8-kp_udp_rgb.py',
    4: 'configs/vitpose-b_8-kp_udp_rgbd.py',
    5: 'configs/vitpose-b_8-kp_udp_rgbdm.py',
}

# Fields allowed to differ between arms. Anything else differing is a bug.
ALLOWED_DIFF_SUBSTRINGS = (
    'in_channels',
    'channels',
    'mean',
    'std',
    'img_mean',
    'img_std',
)


def flatten(obj, prefix=''):
    """Config -> {dotted.path: leaf value}, with list indices in the path."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f'{prefix}.{k}' if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten(v, f'{prefix}[{i}]'))
    else:
        out[prefix] = obj
    return out


def check_config_diff(cfgs):
    print('\n[1/4] config diff  -- arms must differ only in channel fields')
    flat = {c: flatten(cfg.to_dict()) for c, cfg in cfgs.items()}
    keys = set().union(*(set(f) for f in flat.values()))
    # `filename`/`work_dir` style bookkeeping is expected to differ.
    keys = {k for k in keys if not k.startswith(('filename', 'work_dir', 'cfg_'))}

    bad = []
    for k in sorted(keys):
        vals = {c: flat[c].get(k, '<absent>') for c in cfgs}
        if len(set(map(repr, vals.values()))) == 1:
            continue
        if any(s in k for s in ALLOWED_DIFF_SUBSTRINGS):
            print(f'    ok (channel field)  {k}')
            print(f'        ' + '  '.join(f'{c}ch={vals[c]!r}' for c in cfgs))
            continue
        bad.append((k, vals))

    for k, vals in bad:
        print(f'    FAIL  {k}')
        for c, v in vals.items():
            print(f'        {c}ch = {v!r}')
    if bad:
        raise SystemExit(
            f'\n{len(bad)} non-channel field(s) differ between the ablation '
            'arms. The comparison would not isolate depth. Fix the configs.')
    print('    PASS -- arms are identical apart from channel fields')


def check_wiring(cfgs):
    print('\n[2/4] wiring  -- loader channels == preprocessor length == backbone')
    for c, cfg in cfgs.items():
        n_pre = len(cfg.model.data_preprocessor.mean)
        n_std = len(cfg.model.data_preprocessor.std)
        n_bb = cfg.model.backbone.in_channels
        loaders = []
        for name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
            for t in cfg[name].dataset.pipeline:
                if t['type'] == 'LoadRGBDImage':
                    loaders.append((name, t['channels']))
        seen = {n for _, n in loaders}
        if not (seen == {c} and n_pre == n_std == n_bb == c):
            raise SystemExit(
                f'    FAIL {c}ch: loader={sorted(seen)} preprocessor mean={n_pre} '
                f'std={n_std} backbone.in_channels={n_bb}; all must equal {c}')
        print(f'    PASS {c}ch: {len(loaders)} loaders, preprocessor {n_pre}, '
              f'backbone {n_bb}')


def check_step0_equivalence(cfgs, skip_pretrained):
    print('\n[3/4] step-0 equivalence  -- 4/5ch tokens == 3ch tokens on same RGB')
    from mmpose.registry import MODELS

    models = {}
    for c, cfg in cfgs.items():
        bb = dict(cfg.model.backbone)
        if skip_pretrained:
            bb.pop('init_cfg', None)
        elif not Path(bb['init_cfg']['checkpoint']).exists():
            raise SystemExit(
                f"    pretrained checkpoint not found: "
                f"{bb['init_cfg']['checkpoint']}\n"
                '    Produce it with scripts/convert_vit_pretrain.py, or pass '
                '--skip-pretrained to check the plumbing without it.')
        torch.manual_seed(0)          # identical random init across arms
        m = MODELS.build(bb)
        m.init_weights()
        m.eval()
        models[c] = m

    torch.manual_seed(1)
    rgb = torch.randn(2, 3, 256, 256)
    # Deliberately LARGE extra channels: if the zero stem leaked, this shows.
    extra = torch.randn(2, 2, 256, 256) * 50

    with torch.no_grad():
        ref = models[3](rgb)[0]
        for c in (4, 5):
            x = torch.cat([rgb, extra[:, :c - 3]], dim=1)
            d = (models[c](x)[0] - ref).abs().max().item()
            print(f'    {c}ch vs 3ch: max|delta| = {d:.3e}')
            if d != 0.0:
                raise SystemExit(
                    f'    FAIL: {c}ch model is not identical to the 3ch model at '
                    'step 0. extra_stem should be zero-initialised '
                    "(extra_stem_mode='zero').")
    print('    PASS -- extra channels contribute exactly zero at init')
    return models


def check_lr_groups(cfgs):
    print('\n[4/4] learning rate  -- extra_stem must not land in layer 0')
    from mmpose.engine.optim_wrappers.layer_decay_optim_wrapper import \
        get_num_layer_for_vit

    cfg = cfgs[5]
    decay = cfg.optim_wrapper.paramwise_cfg['layer_decay_rate']
    n_layers = cfg.optim_wrapper.paramwise_cfg['num_layers'] + 2

    def scale(name):
        lid = get_num_layer_for_vit(name, n_layers)
        return lid, decay**(n_layers - lid - 1)

    pe_id, pe_scale = scale('backbone.patch_embed.proj.weight')
    es_id, es_scale = scale('backbone.extra_stem.proj.weight')
    b0_id, b0_scale = scale('backbone.layers.0.attn.qkv.weight')
    print(f'    patch_embed  layer {pe_id:2d}  lr_scale {pe_scale:.4f}')
    print(f'    extra_stem   layer {es_id:2d}  lr_scale {es_scale:.4f}')
    print(f'    layers.0     layer {b0_id:2d}  lr_scale {b0_scale:.4f}')
    if b0_scale == 1.0:
        raise SystemExit(
            '    FAIL: transformer blocks got lr_scale 1.0, so layer-wise decay '
            'is a no-op. They must be named `backbone.layers.<i>`.')
    if es_scale < 0.5:
        raise SystemExit(
            f'    FAIL: extra_stem lr_scale is {es_scale:.4f}. A zero-initialised '
            'stem at that LR never leaves zero, and the depth arms would '
            'measure nothing. Keep it out of the `patch_embed` namespace.')
    print('    PASS -- depth stem trains at full LR, blocks decay properly')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-pretrained', action='store_true',
                    help='build backbones without loading the pretrain')
    args = ap.parse_args()

    cfgs = {c: Config.fromfile(p) for c, p in CONFIGS.items()}
    init_default_scope('mmpose')
    print('Loaded:')
    for c, p in CONFIGS.items():
        print(f'    {c}ch  {p}')

    check_config_diff(cfgs)
    check_wiring(cfgs)
    check_step0_equivalence(cfgs, args.skip_pretrained)
    check_lr_groups(cfgs)
    print('\nAll four checks passed. The ablation isolates depth.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
