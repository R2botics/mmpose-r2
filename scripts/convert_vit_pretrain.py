#!/usr/bin/env python3
"""Adapt an MAE or ViTPose checkpoint to this repo's ViT backbone.

Mirrors `scripts/convert_hrnet_to_rgbd.py`: the transformation happens ONCE,
offline, producing a checkpoint you can inspect and diff, rather than hiding
inside a load hook where a silently-skipped tensor looks like a successful run.

Three transformations, none of them optional:

1. **`blocks.` -> `layers.`** — this repo names the transformer blocks `layers`
   so that `LayerDecayOptimWrapperConstructor` recognises them. Skip this and
   every block quietly lands in the constructor's `else` branch at
   `lr_scale = 1.0`, i.e. layer-wise LR decay does nothing at all.

2. **`norm.` -> `last_norm.`** — MAE calls the final LayerNorm `norm`; ViTPose
   calls it `last_norm`. Exact-match only, so the per-block `norm1`/`norm2`
   are untouched.

3. **`pos_embed` resize** — MAE is pretrained at 224x224 (14x14 grid + cls
   token = 197), ViTPose-COCO at 256x192 (16x12 + 1 = 193). This project runs
   256x256 (16x16 + 1 = 257). The patch tokens are bicubically resampled on
   their 2-D grid; the leading class-token row is carried across untouched
   (ViT.forward folds it into every token as a constant offset).

What this script deliberately does NOT do is widen `patch_embed.proj` to 5
input channels. Extra channels are handled at RUNTIME by
`ViT.extra_stem` (a separate zero-initialised projection) — see the rationale
in `mmpose/models/backbones/vit.py`. That keeps ONE pretrained checkpoint
valid for the 3-, 4- and 5-channel ablation runs, so those three runs differ
only by the presence of the extra stem.

Usage:
    # ViT-B, MAE pretrain (the ViTPose paper's starting point)
    python scripts/convert_vit_pretrain.py \
        --src work_dirs/pretrained/mae_pretrain_vit_base.pth \
        --dst work_dirs/pretrained/vit_base_256x256.pth

    # ViT-B, ViTPose COCO-trained weights (non-square source grid)
    python scripts/convert_vit_pretrain.py \
        --src work_dirs/pretrained/vitpose_base_coco_256x192.pth \
        --dst work_dirs/pretrained/vitpose_base_256x256.pth \
        --src-grid 16 12
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

# Checkpoint prefixes that belong to a task head, not the backbone.
DROP_PREFIXES = ('decoder_', 'keypoint_head.', 'head.', 'mask_token',
                 'associate_head.')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True, help='source .pth')
    p.add_argument('--dst', required=True, help='output .pth')
    p.add_argument('--input-size', type=int, nargs=2, default=(256, 256),
                   metavar=('H', 'W'), help='target input size (default 256 256)')
    p.add_argument('--patch-size', type=int, default=16)
    p.add_argument('--ratio', type=int, default=1,
                   help='sub-patch stride factor; 2 => stride 8 tokens')
    p.add_argument('--src-grid', type=int, nargs=2, default=None,
                   metavar=('H', 'W'),
                   help='source pos_embed token grid. Auto-detected when '
                        'square; required otherwise (ViTPose COCO: 16 12).')
    return p.parse_args()


def unwrap(ckpt):
    """Pull the tensor dict out of whatever wrapper the checkpoint uses."""
    for key in ('state_dict', 'model', 'module'):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def normalise_keys(state):
    """Strip wrapper prefixes, drop head weights, apply the two renames."""
    out = {}
    dropped, renamed = [], []
    for k, v in state.items():
        name = k
        for prefix in ('module.', 'backbone.', 'encoder.'):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if any(name.startswith(p) for p in DROP_PREFIXES):
            dropped.append(k)
            continue
        if name.startswith('blocks.'):
            name = 'layers.' + name[len('blocks.'):]
            renamed.append((k, name))
        elif name in ('norm.weight', 'norm.bias'):
            name = 'last_norm.' + name.split('.')[1]
            renamed.append((k, name))
        out[name] = v
    return out, dropped, renamed


def resize_pos_embed(pos_embed, src_grid, dst_grid):
    """Bicubically resample the patch-token grid, keeping extra tokens as-is."""
    n_tokens, dim = pos_embed.shape[-2], pos_embed.shape[-1]
    sh, sw = src_grid
    n_extra = n_tokens - sh * sw
    if n_extra < 0:
        raise ValueError(
            f'pos_embed has {n_tokens} tokens, fewer than the {sh}x{sw}={sh*sw} '
            f'patch grid implied by --src-grid.')

    extra = pos_embed[:, :n_extra]
    patches = pos_embed[:, n_extra:]
    patches = patches.reshape(1, sh, sw, dim).permute(0, 3, 1, 2)
    patches = F.interpolate(
        patches.float(), size=dst_grid, mode='bicubic', align_corners=False)
    patches = patches.permute(0, 2, 3, 1).flatten(1, 2)
    return torch.cat([extra, patches], dim=1)


def infer_src_grid(pos_embed, explicit):
    if explicit is not None:
        return tuple(explicit)
    # Try each plausible count of non-patch tokens (0 or 1 class token).
    for n_extra in (1, 0):
        n = pos_embed.shape[-2] - n_extra
        side = int(round(n**0.5))
        if side * side == n:
            return (side, side)
    raise SystemExit(
        f'pos_embed has {pos_embed.shape[-2]} tokens, which is not a square '
        'grid (+/- a class token). Pass --src-grid H W explicitly; ViTPose '
        'COCO 256x192 checkpoints are 16 12.')


def main():
    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        raise SystemExit(f'source checkpoint not found: {src}')

    ckpt = torch.load(str(src), map_location='cpu', weights_only=False)
    state = unwrap(ckpt)
    print(f'Loaded {src}  ({len(state)} tensors)')

    state, dropped, renamed = normalise_keys(state)
    print(f'  dropped {len(dropped)} head/decoder tensors')
    print(f'  renamed {len(renamed)} block/norm tensors '
          f'(blocks.->layers., norm.->last_norm.)')

    # ---- patch_embed kernel -------------------------------------------------
    if 'patch_embed.proj.weight' in state:
        w = state['patch_embed.proj.weight']
        if w.shape[1] != 3:
            raise SystemExit(
                f'patch_embed.proj.weight has {w.shape[1]} input channels; '
                'this converter expects a 3-channel RGB stem. Extra channels '
                'are added at runtime by ViT.extra_stem, not baked into the '
                'checkpoint.')
        k = args.patch_size
        if w.shape[-1] != k:
            pad = k - w.shape[-1]
            if pad < 0:
                raise SystemExit(
                    f'checkpoint patch kernel {w.shape[-1]} is larger than the '
                    f'requested --patch-size {k}; refusing to crop it.')
            l = pad // 2
            state['patch_embed.proj.weight'] = F.pad(w, (l, pad - l, l, pad - l))
            print(f'  patch kernel zero-padded {w.shape[-1]} -> {k}')

    # ---- pos_embed ----------------------------------------------------------
    if 'pos_embed' not in state:
        raise SystemExit('checkpoint has no pos_embed; is this a ViT?')
    pos = state['pos_embed']
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)
    src_grid = infer_src_grid(pos, args.src_grid)
    dst_grid = (args.input_size[0] // args.patch_size * args.ratio,
                args.input_size[1] // args.patch_size * args.ratio)
    if src_grid == dst_grid:
        print(f'  pos_embed grid already {dst_grid[0]}x{dst_grid[1]}; unchanged')
        state['pos_embed'] = pos
    else:
        state['pos_embed'] = resize_pos_embed(pos, src_grid, dst_grid)
        print(f'  pos_embed resized {src_grid[0]}x{src_grid[1]} -> '
              f'{dst_grid[0]}x{dst_grid[1]}  '
              f'{tuple(pos.shape)} -> {tuple(state["pos_embed"].shape)}')

    # `cls_token` is unused by this backbone (ViT.forward folds pos_embed[:, :1]
    # into every token instead). Left in the file harmlessly; loading is
    # non-strict and it shows up as one unexpected key.
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': state}, str(dst))
    print(f'\nSaved {len(state)} tensors to {dst}')
    print('Use in your config:\n'
          f"    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='{dst}'))")


if __name__ == '__main__':
    main()
