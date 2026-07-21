"""Run RGB UDP, RGBD, and RGBD+Geom models on the same folder, save 4-way viz.

Folder structure expected:
    <input_dir>/
        images/<stem>_cropped_color.png
        depth/<stem>_aligned_depth.png

Outputs:
    <out_dir>/
        rgb/<filename>.png            -- annotated by RGB UDP model
        rgbd/<filename>.png           -- annotated by RGBD model
        rgbd_geom/<filename>.png      -- annotated by RGBD + geometric loss model
        side_by_side/<filename>.png   -- the four panels stitched together
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.registry import init_default_scope

# Register custom transforms / preprocessor / loss (must happen before init_model)
import mmpose.datasets.transforms.rgbd            # noqa: F401
import mmpose.models.data_preprocessors.nchannel  # noqa: F401
import mmpose.models.losses.geometric_loss        # noqa: F401

from mmpose.apis import init_model

RGB_CFG                = 'configs/hrnet-w32_8-kp_udp.py'
RGBD_CFG               = 'configs/hrnet-w32_8-kp_udp_rgbd.py'
RGBD_GEOM_CFG          = 'configs/hrnet-w32_8-kp_udp_rgbd_geom.py'
RGBD_GEOM_SCRATCH_CFG  = 'configs/hrnet-w32_8-kp_udp_rgbd_geom_scratch.py'
RGBD_GEOM_COMBINED_CFG = 'configs/hrnet-w32_8-kp_udp_rgbd_geom_combined.py'


# -- Skeleton drawing -------------------------------------------------------
KPT_COLORS = [
    (255, 153, 51),   # NORTH_1
    (255, 153, 51),   # NORTH_2
    (0, 255, 0),      # EAST_1
    (0, 255, 0),      # EAST_2
    (0, 128, 255),    # SOUTH_1
    (0, 128, 255),    # SOUTH_2
    (255, 51, 255),   # WEST_1
    (255, 51, 255),   # WEST_2
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


# -- Pipeline builders ------------------------------------------------------
def build_pipeline(cfg, with_depth):
    """Build a minimal test pipeline. Uses LoadRGBDImage for RGBD models,
    plain LoadImage otherwise."""
    input_size = cfg.codec.input_size if hasattr(cfg, 'codec') else (256, 256)
    load = dict(type='LoadRGBDImage') if with_depth else dict(type='LoadImage')
    return Compose([
        load,
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=input_size, use_udp=True),
        dict(type='PackPoseInputs'),
    ])


# -- Single-image inference -------------------------------------------------
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
        results = model.test_step(
            {'inputs': inputs, 'data_samples': data_samples})

    pred = results[0].pred_instances
    return pred.keypoints[0], pred.keypoint_scores[0], color


def render_depth(depth_path, ref_color):
    """Turn an aligned uint16 depth image into a TURBO-colormapped viz."""
    out = np.zeros_like(ref_color)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return out
    valid = depth > 0
    if not valid.any():
        return out
    vmin, vmax = depth[valid].min(), depth[valid].max()
    norm = np.zeros_like(depth, dtype=np.uint8)
    norm[valid] = np.clip(
        (depth[valid] - vmin) / max(vmax - vmin, 1) * 255,
        0, 255).astype(np.uint8)
    out = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    out[~valid] = (0, 0, 0)
    return out


# -- Main -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',
                        default='data/test_OOD',
                        help='Folder with images/ and depth/ subfolders')
    parser.add_argument('--image',
                        default=None,
                        help='Path to a single color image. If set, only this '
                             'image is processed (and --input-dir is ignored). '
                             'The matching depth file is found by replacing '
                             '_cropped_color.png with _aligned_depth.png and '
                             'looking in a sibling depth/ folder.')
    parser.add_argument('--rgb-checkpoint',
                        default='work_dirs/hrnet_udp_test_01/best_coco_AP_epoch_40.pth')
    parser.add_argument('--rgbd-checkpoint',
                        default='work_dirs/hrnet_udp_rgbd_test_01/best_coco_AP_epoch_20.pth')
    parser.add_argument('--rgbd-geom-checkpoint',
                        default='work_dirs/hrnet_udp_rgbd_geom_test_01/best_coco_AP_epoch_30.pth')
    parser.add_argument('--rgbd-geom-scratch-checkpoint',
                        default='work_dirs/hrnet_udp_rgbd_geom_scratch_test_01/best_coco_AP_epoch_510.pth')
    parser.add_argument('--rgbd-geom-combined-checkpoint',
                        default='work_dirs/hrnet_udp_rgbd_geom_combined_test_02/best_coco_AP_epoch_30.pth')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    init_default_scope('mmpose')

    out_root = Path(args.out_dir)
    (out_root / 'rgb').mkdir(parents=True, exist_ok=True)
    (out_root / 'rgbd').mkdir(parents=True, exist_ok=True)
    (out_root / 'rgbd_geom').mkdir(parents=True, exist_ok=True)
    (out_root / 'rgbd_geom_scratch').mkdir(parents=True, exist_ok=True)
    (out_root / 'rgbd_geom_combined').mkdir(parents=True, exist_ok=True)
    (out_root / 'side_by_side').mkdir(parents=True, exist_ok=True)

    # Load all three models
    print('Loading RGB model...')
    rgb_cfg = Config.fromfile(RGB_CFG)
    rgb_model = init_model(rgb_cfg, args.rgb_checkpoint, device=args.device)
    rgb_pipeline = build_pipeline(rgb_cfg, with_depth=False)

    print('Loading RGBD model...')
    rgbd_cfg = Config.fromfile(RGBD_CFG)
    rgbd_model = init_model(rgbd_cfg, args.rgbd_checkpoint, device=args.device)
    rgbd_pipeline = build_pipeline(rgbd_cfg, with_depth=True)

    print('Loading RGBD+Geom model...')
    geom_cfg = Config.fromfile(RGBD_GEOM_CFG)
    geom_model = init_model(geom_cfg, args.rgbd_geom_checkpoint, device=args.device)
    geom_pipeline = build_pipeline(geom_cfg, with_depth=True)

    print('Loading RGBD+Geom (from scratch) model...')
    scratch_cfg = Config.fromfile(RGBD_GEOM_SCRATCH_CFG)
    scratch_model = init_model(
        scratch_cfg, args.rgbd_geom_scratch_checkpoint, device=args.device)
    scratch_pipeline = build_pipeline(scratch_cfg, with_depth=True)

    print('Loading RGBD+Geom (combined real+synth) model...')
    combined_cfg = Config.fromfile(RGBD_GEOM_COMBINED_CFG)
    combined_model = init_model(
        combined_cfg, args.rgbd_geom_combined_checkpoint, device=args.device)
    combined_pipeline = build_pipeline(combined_cfg, with_depth=True)

    # Decide whether to process a single image or a whole folder
    if args.image:
        color_paths = [Path(args.image)]
        # input_dir used later to find the depth file by name pattern
        input_dir = color_paths[0].parent.parent
        print(f'Processing single image: {color_paths[0]}')
    else:
        input_dir = Path(args.input_dir)
        color_paths = sorted(
            (input_dir / 'images').glob('*_cropped_color.png'))
        print(f'Processing {len(color_paths)} images from {input_dir}')

    for color_path in color_paths:
        # 3 model inferences on the same color image
        rgb_kpts, rgb_scores, color = run_inference(
            rgb_model, rgb_pipeline, color_path)
        rgb_viz = draw_pose(color, rgb_kpts, rgb_scores, args.kpt_thr)

        rgbd_kpts, rgbd_scores, _ = run_inference(
            rgbd_model, rgbd_pipeline, color_path)
        rgbd_viz = draw_pose(color, rgbd_kpts, rgbd_scores, args.kpt_thr)

        geom_kpts, geom_scores, _ = run_inference(
            geom_model, geom_pipeline, color_path)
        geom_viz = draw_pose(color, geom_kpts, geom_scores, args.kpt_thr)

        scratch_kpts, scratch_scores, _ = run_inference(
            scratch_model, scratch_pipeline, color_path)
        scratch_viz = draw_pose(color, scratch_kpts, scratch_scores, args.kpt_thr)

        combined_kpts, combined_scores, _ = run_inference(
            combined_model, combined_pipeline, color_path)
        combined_viz = draw_pose(color, combined_kpts, combined_scores, args.kpt_thr)

        # Depth visualization
        depth_path = (input_dir / 'depth' /
                      color_path.name.replace('_cropped_color.png',
                                              '_aligned_depth.png'))
        depth_viz = render_depth(depth_path, color)

        # 6-panel side-by-side: RGB | RGBD | Geom | Geom+Synth | Geom(scratch) | Depth
        h, w = color.shape[:2]
        gap = np.full((h, 10, 3), 255, dtype=np.uint8)
        sbs = np.concatenate(
            [rgb_viz, gap, rgbd_viz, gap, geom_viz, gap,
             combined_viz, gap, scratch_viz, gap, depth_viz], axis=1)

        # Header with labels positioned over each panel
        header = np.full((30, sbs.shape[1], 3), 50, dtype=np.uint8)
        labels = [
            ('RGB UDP',           10),
            ('RGBD',              w + 20),
            ('RGBD+Geom',         2 * w + 30),
            ('RGBD+Geom+Synth',   3 * w + 40),
            ('Geom (scratch)',    4 * w + 50),
            ('Depth (norm)',      5 * w + 60),
        ]
        for text, x in labels:
            cv2.putText(header, text, (x, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        sbs = np.concatenate([header, sbs], axis=0)

        cv2.imwrite(str(out_root / 'rgb'                / color_path.name), rgb_viz)
        cv2.imwrite(str(out_root / 'rgbd'               / color_path.name), rgbd_viz)
        cv2.imwrite(str(out_root / 'rgbd_geom'          / color_path.name), geom_viz)
        cv2.imwrite(str(out_root / 'rgbd_geom_scratch'  / color_path.name), scratch_viz)
        cv2.imwrite(str(out_root / 'rgbd_geom_combined' / color_path.name), combined_viz)
        cv2.imwrite(str(out_root / 'side_by_side'       / color_path.name), sbs)

    print(f'Done. Output in {out_root}/')
    print(f'  rgb/                  - RGB UDP predictions')
    print(f'  rgbd/                 - RGBD predictions')
    print(f'  rgbd_geom/            - RGBD+Geom (real-only) predictions')
    print(f'  rgbd_geom_combined/   - RGBD+Geom (real+synth) predictions')
    print(f'  rgbd_geom_scratch/    - RGBD+Geom (from scratch) predictions')
    print(f'  side_by_side/         - all six panels stitched (5 models + depth)')


if __name__ == '__main__':
    main()
