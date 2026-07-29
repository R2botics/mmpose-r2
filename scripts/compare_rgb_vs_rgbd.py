"""Run any number of pose models on a folder of images and produce a
side-by-side visualization comparison.

Models are passed as one or more `--model CONFIG CHECKPOINT LABEL` triples.
The script infers whether each model needs RGBD input from the number of
channels in its data preprocessor (5 -> RGBD+mask, 4 -> RGBD, 3 -> RGB-only).

Folder structure expected:
    <input_dir>/
        images/<stem>_cropped_color.png
        depth/<stem>_aligned_depth.png     (optional; only needed for RGBD models)
        annotations/<something>.json       (optional; only for --ann-file scoring)

Outputs:
    <out_dir>/
        <label>/<filename>.png             -- one folder per model
        depth/<filename>.png               -- depth colormap for reference
        side_by_side/<filename>.png        -- all panels + depth stitched
        per_keypoint_errors.csv            -- only with --ann-file

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

Scoring against ground truth:
    Pass `--ann-file` to additionally measure how far each prediction lands
    from the labelled keypoint. This prints a per-model summary, writes a
    per-keypoint CSV, and draws the ground truth on each panel as a hollow
    ring joined to the prediction by an error vector.

    python scripts/compare_rgb_vs_rgbd.py \\
        --input-dir data/OOD \\
        --ann-file data/OOD/annotations/person_keypoints_Test.json \\
        --out-dir work_dirs/ood_scored \\
        --model <config> <checkpoint> Finetune
"""
import argparse
import csv
import json
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


def draw_gt_overlay(img, pred_kpts, gt_kpts):
    """Draw ground truth as hollow rings, joined to predictions by an error
    vector. `gt_kpts` is (K, 3) COCO-style (x, y, visibility)."""
    out = img
    for i, (gx, gy, v) in enumerate(gt_kpts):
        if v <= 0:                       # unlabelled -- nothing to compare to
            continue
        gt_pt = (int(gx), int(gy))
        pred_pt = (int(pred_kpts[i][0]), int(pred_kpts[i][1]))
        cv2.line(out, gt_pt, pred_pt, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.circle(out, gt_pt, 5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(out, gt_pt, 4, KPT_COLORS[i], 1, cv2.LINE_AA)
    return out


# -- Ground truth / error metrics -------------------------------------------
def load_ground_truth(ann_file):
    """Read a COCO keypoint JSON.

    Returns (gt_by_filename, keypoint_names) where each value is an (K, 3)
    float array of (x, y, visibility) in original-image pixel coordinates --
    the same frame the model's predictions come back in.
    """
    with open(ann_file) as f:
        data = json.load(f)

    id_to_name = {img['id']: img['file_name'] for img in data['images']}
    gt = {}
    for ann in data['annotations']:
        kpts = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
        name = id_to_name.get(ann['image_id'])
        if name is not None:
            # One annotation per image in this dataset; last one wins.
            gt[name] = kpts

    names = None
    for cat in data.get('categories', []):
        if cat.get('keypoints'):
            names = list(cat['keypoints'])
            break
    return gt, names


def keypoint_errors(pred_kpts, gt_kpts):
    """Per-keypoint Euclidean distance in pixels.

    Returns a (K,) array holding NaN wherever the keypoint is unlabelled, so
    downstream aggregation with nan-aware reductions skips it rather than
    silently scoring an invisible point as a perfect hit.
    """
    dists = np.full(len(gt_kpts), np.nan, dtype=np.float64)
    visible = gt_kpts[:, 2] > 0
    if visible.any():
        deltas = pred_kpts[visible][:, :2] - gt_kpts[visible, :2]
        dists[visible] = np.linalg.norm(deltas, axis=1)
    return dists


def summarize_errors(errors, norms, pck_thr, kpt_names):
    """Aggregate per-image error rows into headline + per-keypoint stats.

    `errors` is a list of (K,) pixel-distance arrays and `norms` the matching
    per-image normalisers (image diagonal). Normalising matters here because
    the OOD frames are individually cropped and range from ~245px to ~412px
    across, so raw pixel error is not comparable frame to frame.
    """
    err = np.vstack(errors)                        # (N, K)
    norm = np.asarray(norms, dtype=np.float64)[:, None]
    normed = err / norm

    with np.errstate(invalid='ignore'):
        headline = dict(
            n_images=len(errors),
            n_points=int(np.sum(~np.isnan(err))),
            mean_px=float(np.nanmean(err)),
            median_px=float(np.nanmedian(err)),
            p90_px=float(np.nanpercentile(err, 90)),
            max_px=float(np.nanmax(err)),
            nme_pct=float(np.nanmean(normed) * 100),
            pck=float(np.nanmean(normed <= pck_thr) * 100),
        )
        per_kpt = []
        for i in range(err.shape[1]):
            col, ncol = err[:, i], normed[:, i]
            if np.all(np.isnan(col)):
                continue
            per_kpt.append(dict(
                name=kpt_names[i] if kpt_names and i < len(kpt_names)
                else f'kpt_{i}',
                mean_px=float(np.nanmean(col)),
                median_px=float(np.nanmedian(col)),
                nme_pct=float(np.nanmean(ncol) * 100),
                pck=float(np.nanmean(ncol <= pck_thr) * 100),
            ))
    return headline, per_kpt


def print_scores(label, headline, per_kpt, pck_thr):
    print(f'\n--- {label} '.ljust(72, '-'))
    print(f'  images scored      : {headline["n_images"]}  '
          f'({headline["n_points"]} labelled keypoints)')
    print(f'  mean error         : {headline["mean_px"]:.2f} px')
    print(f'  median error       : {headline["median_px"]:.2f} px')
    print(f'  p90 / max error    : {headline["p90_px"]:.2f} / '
          f'{headline["max_px"]:.2f} px')
    print(f'  NME (% of diag)    : {headline["nme_pct"]:.2f} %')
    print(f'  {f"PCK@{pck_thr:g}":<19s}: {headline["pck"]:.1f} %')
    print(f'  {"keypoint":<12s}{"mean px":>10s}{"median px":>12s}'
          f'{"NME %":>9s}{"PCK %":>8s}')
    for k in per_kpt:
        print(f'  {k["name"]:<12s}{k["mean_px"]:>10.2f}{k["median_px"]:>12.2f}'
              f'{k["nme_pct"]:>9.2f}{k["pck"]:>8.1f}')


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
    parser.add_argument('--ann-file', default=None,
                        help='COCO keypoint JSON. If given, score predictions '
                             'against ground truth and write a CSV.')
    parser.add_argument('--pck-thr', type=float, default=0.05,
                        help='PCK threshold as a fraction of the image '
                             'diagonal (default 0.05)')
    parser.add_argument('--no-gt-overlay', action='store_true',
                        help='Do not draw ground truth on the output panels')
    parser.add_argument('--model', nargs=3, action='append', required=True,
                        metavar=('CONFIG', 'CHECKPOINT', 'LABEL'),
                        help='Add a model to compare. Repeat this flag '
                             'once per model.')
    args = parser.parse_args()

    init_default_scope('mmpose')

    gt_by_name, kpt_names = ({}, None)
    if args.ann_file:
        gt_by_name, kpt_names = load_ground_truth(args.ann_file)
        print(f'Loaded ground truth for {len(gt_by_name)} images '
              f'from {args.ann_file}')

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
            'errors': [],   # one (K,) array of pixel distances per image
            'norms': [],    # matching per-image normaliser (image diagonal)
        })

    csv_rows = []

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

        gt_kpts = gt_by_name.get(color_path.name)
        if args.ann_file and gt_kpts is None:
            print(f'  WARNING: no ground-truth entry for {color_path.name}; '
                  f'skipping it in the score')

        # Run inference for each model
        panels = []
        for m in models:
            kpts, scores, color = run_inference(
                m['model'], m['pipeline'], color_path)
            viz = draw_pose(color, kpts, scores, args.kpt_thr)

            if gt_kpts is not None:
                # Normalise by image diagonal: the OOD crops differ in size,
                # so raw pixels are not comparable across frames.
                h, w = color.shape[:2]
                diag = float(np.hypot(w, h))
                dists = keypoint_errors(kpts, gt_kpts)
                m['errors'].append(dists)
                m['norms'].append(diag)
                for i, d in enumerate(dists):
                    if np.isnan(d):
                        continue
                    csv_rows.append(dict(
                        image=color_path.name,
                        model=m['label'],
                        keypoint=(kpt_names[i] if kpt_names and i < len(kpt_names)
                                  else f'kpt_{i}'),
                        error_px=round(float(d), 4),
                        error_norm=round(float(d / diag), 6),
                        score=round(float(scores[i]), 4),
                    ))
                if not args.no_gt_overlay:
                    viz = draw_gt_overlay(viz, kpts, gt_kpts)
                mean_err = np.nanmean(dists)
                if not np.isnan(mean_err):
                    cv2.putText(viz, f'{mean_err:.1f}px', (5, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1, cv2.LINE_AA)

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

    # -- Scoring summary ----------------------------------------------------
    if csv_rows:
        csv_path = out_root / 'per_keypoint_errors.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f'\n{"=" * 72}')
        print(f'Error vs ground truth  (PCK threshold = {args.pck_thr:g} '
              f'x image diagonal)')
        print('=' * 72)
        for m in models:
            if not m['errors']:
                continue
            headline, per_kpt = summarize_errors(
                m['errors'], m['norms'], args.pck_thr, kpt_names)
            print_scores(m['label'], headline, per_kpt, args.pck_thr)
        print(f'\n  per-keypoint CSV: {csv_path}')

    print(f'\nDone. Output in {out_root}/')
    for m in models:
        print(f'  {m["folder"]:<30s} - {m["label"]} predictions')
    print(f'  depth                          - depth colormap reference')
    print(f'  side_by_side                   - all panels stitched')
    if csv_rows:
        print(f'  per_keypoint_errors.csv        - per-keypoint distances')


if __name__ == '__main__':
    main()
