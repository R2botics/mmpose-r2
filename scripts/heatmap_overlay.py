"""Overlay predicted heatmaps on the model's INPUT image, per keypoint.

Everything is rendered in the network's own 256x256 input space, so no inverse
affine is involved -- the heatmap and the pixels it was computed from are
guaranteed to be registered.

What to look for:
  * peak leaning toward the box interior      -> the inboard bias, visible
  * two peaks (rim and inner wall / contents) -> bimodal, confidence can't see it
  * peak on a contents edge                   -> locked onto the wrong structure
  * broad low blob                            -> no evidence, decode is a guess

Usage:
    python scripts/heatmap_overlay.py <config> <ckpt> <image.png> [-o out.png]
    python scripts/heatmap_overlay.py <config> <ckpt> <dir> --glob '*.png' -o outdir
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_rgb_vs_rgbd import load_model  # noqa: E402

NAMES = ['NORTH_1', 'NORTH_2', 'EAST_1', 'EAST_2',
         'SOUTH_1', 'SOUTH_2', 'WEST_1', 'WEST_2']


def build_sample(path):
    color = cv2.imread(str(path))
    h, w = color.shape[:2]
    return dict(
        img_path=str(path), bbox=np.array([[0, 0, w, h]], np.float32),
        bbox_score=np.array([1.], np.float32),
        bbox_scale=np.array([[w, h]], np.float32),
        bbox_center=np.array([[w / 2, h / 2]], np.float32),
        id=0, category_id=1,
        keypoints=np.zeros((1, 8, 2), np.float32),
        keypoints_visible=np.zeros((1, 8), np.float32),
        img_shape=(h, w), ori_shape=(h, w))


def warp_gt(geo_pipeline, path, gt_kpts):
    """Ground truth in the network's 256x256 input space.

    TopdownAffine writes the warped points to `transformed_keypoints` and
    leaves `keypoints` in ORIGINAL image coordinates -- reading the wrong key
    silently gives you untransformed points that look plausible.
    """
    if geo_pipeline is None or gt_kpts is None:
        return None
    res = build_sample(path)
    res['keypoints'] = gt_kpts[None, :, :2].astype(np.float32).copy()
    res['keypoints_visible'] = gt_kpts[None, :, 2].astype(np.float32).copy()
    for t in geo_pipeline:
        res = t(res)
    warped = res.get('transformed_keypoints')
    if warped is None or warped.ndim < 3:
        return None
    return warped[0]


def render(model, pipeline, path, out_path, geo_pipeline=None, gt_kpts=None):
    data = pipeline(build_sample(path))
    gt_in = warp_gt(geo_pipeline, path, gt_kpts)
    inputs = data['inputs'].unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        # MUST go through the data preprocessor: it normalises RGB and rescales
        # the uint16 depth channel. Feeding the raw pipeline tensor straight to
        # the backbone produces meaningless heatmaps that look like a model bug.
        batch = model.data_preprocessor(
            {'inputs': inputs, 'data_samples': [data['data_samples']]},
            training=False)
        heat = model.head.forward(
            model.extract_feat(batch['inputs']))[0].cpu().numpy()

    # inputs are pre-normalisation (the data preprocessor normalises), so the
    # first three channels are the warped RGB in 0-255.
    rgb = data['inputs'][:3].numpy().transpose(1, 2, 0)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    size = rgb.shape[:2][::-1]

    tiles = []
    for k in range(8):
        hm = heat[k]
        peak = np.unravel_index(hm.argmax(), hm.shape)
        big = cv2.resize(hm, size, interpolation=cv2.INTER_CUBIC)
        norm = (big - big.min()) / max(big.max() - big.min(), 1e-9)
        cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        tile = cv2.addWeighted(rgb, 0.55, cmap, 0.45, 0)

        # mark the argmax, in input-space pixels
        py = peak[0] * (size[1] - 1) / (hm.shape[0] - 1)
        px = peak[1] * (size[0] - 1) / (hm.shape[1] - 1)
        cv2.drawMarker(tile, (int(px), int(py)), (255, 255, 255),
                       cv2.MARKER_CROSS, 18, 2)

        label = f'{NAMES[k]}  max{hm.max():.2f}'
        if gt_in is not None and gt_kpts is not None and gt_kpts[k, 2] > 0:
            gx, gy = float(gt_in[k, 0]), float(gt_in[k, 1])
            # hollow circle = ground truth, cross = heatmap argmax
            cv2.circle(tile, (int(gx), int(gy)), 8, (0, 255, 255), 2)
            cv2.line(tile, (int(gx), int(gy)), (int(px), int(py)),
                     (0, 255, 255), 1, cv2.LINE_AA)
            label += f'  off{np.hypot(gx - px, gy - py):.0f}'
        cv2.putText(tile, label, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(tile)

    grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    cv2.imwrite(str(out_path), grid)
    return {NAMES[k]: float(heat[k].max()) for k in range(8)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('target', help='image file or directory')
    p.add_argument('-o', '--out', required=True)
    p.add_argument('--glob', default='*.png')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--ann-file', default=None,
                   help='COCO JSON. If given, each output is prefixed with the '
                        'image rank by max keypoint error, so the worst cases '
                        'sort to the top of the folder.')
    args = p.parse_args()

    model, pipeline, _, cfg = load_model(args.config, args.checkpoint, args.device)

    # Geometry-only pipeline: stops before GenerateTarget/PackPoseInputs so the
    # warped ground truth is still available as `transformed_keypoints`.
    from mmpose.registry import TRANSFORMS
    geo_pipeline = [
        TRANSFORMS.build(dict(t)) for t in cfg.val_dataloader.dataset.pipeline
        if t['type'] in ('LoadRGBDImage', 'GetBBoxCenterScale', 'TopdownAffine')
    ]
    target = Path(args.target)
    if target.is_dir():
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        gt = {}
        if args.ann_file:
            from compare_rgb_vs_rgbd import load_ground_truth, run_inference
            gt, _ = load_ground_truth(args.ann_file)

        images = sorted(target.glob(args.glob))
        scored = []
        for img in images:
            g = gt.get(img.name)
            if g is None:
                scored.append((np.nan, np.nan, img))
                continue
            pred = np.asarray(run_inference(model, pipeline, img)[0])[:, :2]
            m = g[:, 2] > 0
            if not m.any():
                scored.append((np.nan, np.nan, img))
                continue
            d = np.linalg.norm(pred[m] - g[m, :2], axis=1)
            scored.append((float(d.max()), float(d.mean()), img))

        # worst first, so the filenames sort in review order
        order = sorted(scored, key=lambda t: (-t[0]) if np.isfinite(t[0]) else 1)
        print(f'{"rank":>5}{"max px":>9}{"mean px":>9}  image')
        for rank, (mx, mn, img) in enumerate(order):
            tag = f'{rank:03d}_' if gt else ''
            peaks = render(model, pipeline, img, out_dir / f'{tag}heat_{img.name}',
                           geo_pipeline, gt.get(img.name))
            lo = min(peaks, key=peaks.get)
            s_mx = f'{mx:9.1f}' if np.isfinite(mx) else f'{"--":>9}'
            s_mn = f'{mn:9.1f}' if np.isfinite(mn) else f'{"--":>9}'
            print(f'{rank:>5}{s_mx}{s_mn}  {img.name[:52]:<54}'
                  f'weakest {lo} {peaks[lo]:.2f}')
    else:
        gt_one = None
        if args.ann_file:
            from compare_rgb_vs_rgbd import load_ground_truth
            gt_one = load_ground_truth(args.ann_file)[0].get(target.name)
        peaks = render(model, pipeline, target, args.out, geo_pipeline, gt_one)
        for n, v in peaks.items():
            print(f'  {n:<10}{v:>7.3f}')
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
