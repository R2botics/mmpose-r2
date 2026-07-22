"""Run any number of pose models on a folder of images and produce a
side-by-side visualization comparison.

Models are passed as one or more `--model CONFIG CHECKPOINT LABEL` triples.
The script infers whether each model needs RGBD input from the number of
channels in its data preprocessor (5 -> RGBD+mask, 4 -> RGBD, 3 -> RGB-only).

Folder structure expected:
    <input_dir>/
        images/<stem>_cropped_color.png
        depth/<stem>_aligned_depth.png     (optional; only needed for RGBD models)

Outputs:
    <out_dir>/
        <label>/<filename>.png             -- one folder per model
        depth/<filename>.png               -- depth colormap for reference
        side_by_side/<filename>.png        -- all panels + depth stitched

Example:
    python scripts/compare_rgb_vs_rgbd.py \\
        --input-dir data/test_OOD \\
        --out-dir ~/Documents/finetune_vs_combined \\
        --kpt-thr 0.5 \\
        --model configs/hrnet-w32_8-kp_udp_rgbd_geom_finetune_real.py \\
                work_dirs/hrnet_synth_then_real_test_01/best_coco_AP_epoch_20.pth \\
                Finetune \\
        --model configs/hrnet-w32_8-kp_udp_rgbd_geom_combined.py \\
                work_dirs/hrnet_udp_rgbd_geom_combined_test_02/best_coco_AP_epoch_30.pth \\
                Combined
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.registry import init_default_scope

# Register custom transforms / preprocessor / loss (needed at model init)
import mmpose.datasets.transforms.rgbd            # noqa: F401
import mmpose.models.data_preprocessors.nchannel  # noqa: F401
import mmpose.models.losses.geometric_loss        # noqa: F401

from mmpose.apis import init_model


# -- Skeleton drawing -------------------------------------------------------
KPT_COLORS = [
    (255, 153, 51),  # NORTH_1
    (255, 153, 51),  # NORTH_2
    (0, 255, 0),     # EAST_1
    (0, 255, 0),     # EAST_2
    (0, 128, 255),   # SOUTH_1
    (0, 128, 255),   # SOUTH_2
    (255, 51, 255),  # WEST_1
    (255, 51, 255),  # WEST_2
]
SKELETON = [(0, 1), (2, 3), (4, 5), (6, 7)]


def draw_pose(img, keypoints, scores, kpt_thr=0.3):
    out = img.copy()
    for (a, b) in SKELETON:
        if scores[a] >= kpt_thr and scores[b] >= kpt_thr:
            pa = tuple(int(x) for x in keypoints[a])
            pb = tuple(int(x) for x in keypoints[b])
            cv2.line(out, pa, pb, (255, 255, 255), 2)
    for i, (x, y) in enumerate(keypoints):
        if scores[i] >= kpt_thr:
            cv2.circle(out, (int(x), int(y)), 4, KPT_COLORS[i], -1)
    return out


# -- Pipeline / inference ---------------------------------------------------
def build_pipeline(cfg, with_depth):
    input_size = cfg.codec.input_size if hasattr(cfg, 'codec') else (256, 256)
    load = dict(type='LoadRGBDImage') if with_depth else dict(type='LoadImage')
    return Compose([
        load,
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=input_size, use_udp=True),
        dict(type='PackPoseInputs'),
    ])


def run_inference(model, pipeline, color_path):
    color = cv2.imread(str(color_path))
    h, w = color.shape[:2]
    sample = {
        'img_path': str(color_path),
        'bbox': np.array([[0, 0, w, h]], dtype=np.float32),
        'bbox_score': np.array([1.0], dtype=np.float32),
        'bbox_scale': np.array([[w, h]], dtype=np.float32),
        'bbox_center': np.array([[w / 2, h / 2]], dtype=np.float32),
        'id': 0,
        'category_id': 1,
        'keypoints': np.zeros((1, 8, 2), dtype=np.float32),
        'keypoints_visible': np.zeros((1, 8), dtype=np.float32),
        'img_shape': (h, w),
        'ori_shape': (h, w),
    }
    data = pipeline(sample)
    inputs = data['inputs'].unsqueeze(0).to(next(model.parameters()).device)
    data_samples = [data['data_samples']]
    with torch.no_grad():
        results = model.test_step({'inputs': inputs, 'data_samples': data_samples})
    pred = results[0].pred_instances
    return pred.keypoints[0], pred.keypoint_scores[0], color


def render_depth(depth_path, ref_color):
    out = np.zeros_like(ref_color)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return out
    valid = depth > 0
    if not valid.any():
        return out
    vmin, vmax = depth[valid].min(), depth[valid].max()
    norm = np.zeros_like(depth, dtype=np.uint8)
    norm[valid] = np.clip((depth[valid] - vmin) / max(vmax - vmin, 1) * 255,
                          0, 255).astype(np.uint8)
    out = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    out[~valid] = (0, 0, 0)
    return out


def sanitize_folder_name(label):
    """Turn a display label into a filesystem-safe folder name."""
    return re.sub(r'[^A-Za-z0-9_-]', '_', label).strip('_') or 'model'


def load_model(cfg_path, ckpt_path, device):
    """Load a model and return (model, pipeline, needs_depth, cfg)."""
    cfg = Config.fromfile(cfg_path)
    model = init_model(cfg, ckpt_path, device=device)
    mean = cfg.model.data_preprocessor.mean
    needs_depth = len(mean) >= 4
    pipeline = build_pipeline(cfg, with_depth=needs_depth)
    return model, pipeline, needs_depth, cfg


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('--input-dir', default='data/test_OOD',
                        help='Folder with images/ and depth/ subfolders')
    parser.add_argument('--image', default=None,
                        help='Optional: path to a single color image instead '
                             'of processing the whole input-dir')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model', nargs=3, action='append', required=True,
                        metavar=('CONFIG', 'CHECKPOINT', 'LABEL'),
                        help='Add a model to compare. Repeat this flag '
                             'once per model.')
    args = parser.parse_args()

    init_default_scope('mmpose')

    out_root = Path(args.out_dir)
    (out_root / 'side_by_side').mkdir(parents=True, exist_ok=True)
    (out_root / 'depth').mkdir(parents=True, exist_ok=True)

    # Load each model
    models = []
    for cfg_path, ckpt_path, label in args.model:
        folder = sanitize_folder_name(label)
        print(f'Loading model "{label}"')
        print(f'  config:     {cfg_path}')
        print(f'  checkpoint: {ckpt_path}')
        model, pipeline, needs_depth, cfg = load_model(
            cfg_path, ckpt_path, args.device)
        print(f'  input channels: '
              f'{len(cfg.model.data_preprocessor.mean)} '
              f'(RGBD={needs_depth})')
        (out_root / folder).mkdir(parents=True, exist_ok=True)
        models.append({
            'label': label,
            'folder': folder,
            'model': model,
            'pipeline': pipeline,
            'needs_depth': needs_depth,
        })

    # Decide input list
    if args.image:
        color_paths = [Path(args.image)]
        input_dir = color_paths[0].parent.parent
        print(f'\nProcessing single image: {color_paths[0]}')
    else:
        input_dir = Path(args.input_dir)
        color_paths = sorted((input_dir / 'images').glob('*_cropped_color.png'))
        print(f'\nProcessing {len(color_paths)} images from {input_dir}')

    for color_path in color_paths:
        depth_path = (input_dir / 'depth' /
                      color_path.name.replace('_cropped_color.png',
                                              '_aligned_depth.png'))

        # Run inference for each model
        panels = []
        for m in models:
            kpts, scores, color = run_inference(
                m['model'], m['pipeline'], color_path)
            viz = draw_pose(color, kpts, scores, args.kpt_thr)
            panels.append((m['label'], m['folder'], viz))
            cv2.imwrite(str(out_root / m['folder'] / color_path.name), viz)

        # Depth panel (uses the last color loaded — same image)
        depth_viz = render_depth(depth_path, color)
        cv2.imwrite(str(out_root / 'depth' / color_path.name), depth_viz)

        # Side-by-side: [model_1] gap [model_2] gap ... gap [depth]
        h, w = color.shape[:2]
        gap = np.full((h, 10, 3), 255, dtype=np.uint8)
        pieces = []
        for _, _, viz in panels:
            pieces.extend([viz, gap])
        pieces.append(depth_viz)
        sbs = np.concatenate(pieces, axis=1)

        # Header with a label centered above each panel
        header = np.full((30, sbs.shape[1], 3), 50, dtype=np.uint8)
        panel_labels = [label for label, _, _ in panels] + ['Depth (norm)']
        for i, text in enumerate(panel_labels):
            x = i * (w + 10) + 10
            cv2.putText(header, text, (x, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        sbs = np.concatenate([header, sbs], axis=0)

        cv2.imwrite(str(out_root / 'side_by_side' / color_path.name), sbs)

    print(f'\nDone. Output in {out_root}/')
    for m in models:
        print(f'  {m["folder"]:<30s} - {m["label"]} predictions')
    print(f'  depth                          - depth colormap reference')
    print(f'  side_by_side                   - all panels stitched')


if __name__ == '__main__':
    main()
