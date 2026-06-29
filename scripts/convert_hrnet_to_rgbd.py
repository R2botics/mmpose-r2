"""Convert HRNet-W32 pretrained weights from 3-channel input to 5-channel.

The original first conv has weight shape [64, 3, 3, 3] (out, in, k, k).
We expand it to [64, 5, 3, 3]:
  - Channels 0,1,2: copy original RGB weights (preserve learned features)
  - Channels 3,4:   initialize from mean(RGB) weights so the depth and mask
                    channels start with a sensible visual prior rather than
                    random noise. This is the standard inflation trick.

Usage:
    python scripts/convert_hrnet_to_rgbd.py
"""
import os
from pathlib import Path

import torch
from mmengine.utils import mkdir_or_exist
from torch.hub import download_url_to_file

SOURCE_URL = (
    'https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth'
)
OUTPUT_DIR = Path('work_dirs/pretrained')
SOURCE_PATH = OUTPUT_DIR / 'hrnet_w32-36af842e.pth'
TARGET_PATH = OUTPUT_DIR / 'hrnet_w32_rgbd.pth'

NUM_NEW_CHANNELS = 5  # R, G, B, depth, mask


def main():
    mkdir_or_exist(OUTPUT_DIR)

    # Step 1: download original if not cached
    if not SOURCE_PATH.exists():
        print(f'Downloading {SOURCE_URL} to {SOURCE_PATH}')
        download_url_to_file(SOURCE_URL, str(SOURCE_PATH))
    else:
        print(f'Using cached source: {SOURCE_PATH}')

    # Step 2: load
    ckpt = torch.load(str(SOURCE_PATH), map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)

    # Step 3: find first conv weight (3 input channels, 4D)
    first_conv_key = None
    for k, v in state_dict.items():
        if v.dim() == 4 and v.shape[1] == 3 and 'conv1' in k:
            first_conv_key = k
            break

    if first_conv_key is None:
        raise RuntimeError(
            'Could not find the first 3-channel conv weight. '
            f'Top-level keys: {list(state_dict.keys())[:10]}')

    print(f'Inflating "{first_conv_key}" from 3 to {NUM_NEW_CHANNELS} input channels')

    w = state_dict[first_conv_key]
    print(f'  Original shape: {tuple(w.shape)}')

    # Step 4: expand to 5 channels
    rgb_mean = w.mean(dim=1, keepdim=True)        # [out, 1, k, k]
    n_extra = NUM_NEW_CHANNELS - 3                # 2 extras: depth + mask
    extras = rgb_mean.repeat(1, n_extra, 1, 1)    # [out, n_extra, k, k]
    w_new = torch.cat([w, extras], dim=1)         # [out, 5, k, k]
    print(f'  New shape:      {tuple(w_new.shape)}')

    state_dict[first_conv_key] = w_new

    # Step 5: save
    if 'state_dict' in ckpt:
        ckpt['state_dict'] = state_dict
    else:
        ckpt = state_dict
    torch.save(ckpt, str(TARGET_PATH))
    print(f'\nSaved 5-channel checkpoint to: {TARGET_PATH}')
    print(f'Use this in your config:  checkpoint="{TARGET_PATH}"')


if __name__ == '__main__':
    main()
