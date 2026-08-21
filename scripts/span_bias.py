"""Measure the error component the geometric loss cannot see.

The perpendicularity and parallelism terms normalise their edge vectors, so
they constrain edge DIRECTION and are blind to edge LENGTH. This script splits
each keypoint's error into the component ALONG its flap edge (which changes the
measured span and was historically unpenalised) and the component ACROSS it,
and reports the signed span error per flap.

Baseline, hrnet-256 chewy2-ep120 on data/OOD, BEFORE the span term existed:

    ALONG  mean 4.08px   ACROSS mean 2.95px   ratio 1.39x
    SOUTH_1 ratio 2.48x  (worst)
    signed span error: NORTH -3.98px (80% too short), SOUTH -6.90px (80% short)

A working span term should pull the ALONG/ACROSS ratio toward 1.0 and the
signed span error toward 0. Median pixel error is NOT the metric to judge it
on -- it acts on a direction that median error barely reflects.

Usage:
    python scripts/span_bias.py <config> <checkpoint> [--input-dir data/OOD] \
        [--ann-file data/OOD/annotations/person_keypoints_Test.json]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_rgb_vs_rgbd import (load_ground_truth, load_model,  # noqa: E402
                                 run_inference)

# keypoint index -> (partner index forming its flap edge, flap name)
EDGE = {0: (1, 'NORTH'), 1: (0, 'NORTH'), 2: (3, 'EAST'), 3: (2, 'EAST'),
        4: (5, 'SOUTH'), 5: (4, 'SOUTH'), 6: (7, 'WEST'), 7: (6, 'WEST')}
NAMES = ['NORTH_1', 'NORTH_2', 'EAST_1', 'EAST_2',
         'SOUTH_1', 'SOUTH_2', 'WEST_1', 'WEST_2']
FLAPS = [('NORTH', (0, 1)), ('EAST', (2, 3)),
         ('SOUTH', (4, 5)), ('WEST', (6, 7))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--input-dir', default='data/OOD')
    p.add_argument('--ann-file',
                   default='data/OOD/annotations/person_keypoints_Test.json')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    gt_all, _ = load_ground_truth(args.ann_file)
    model, pipeline, _, _ = load_model(args.config, args.checkpoint, args.device)

    along, across = [], []
    per_kpt = {n: [[], []] for n in NAMES}
    span_err = {n: [] for n, _ in FLAPS}

    images = sorted((Path(args.input_dir) / 'images').glob('*.png'))
    for img_path in images:
        g = gt_all.get(img_path.name)
        if g is None:
            continue
        pred = np.asarray(run_inference(model, pipeline, img_path)[0])[:, :2]

        for i in range(8):
            j, _ = EDGE[i]
            if g[i, 2] <= 0 or g[j, 2] <= 0:
                continue
            edge = g[j, :2] - g[i, :2]
            length = np.linalg.norm(edge)
            if length < 1e-6:
                continue
            unit = edge / length
            normal = np.array([-unit[1], unit[0]])
            err = pred[i] - g[i, :2]
            a, c = abs(float(err @ unit)), abs(float(err @ normal))
            along.append(a)
            across.append(c)
            per_kpt[NAMES[i]][0].append(a)
            per_kpt[NAMES[i]][1].append(c)

        for name, (i, j) in FLAPS:
            if g[i, 2] > 0 and g[j, 2] > 0:
                span_err[name].append(
                    np.linalg.norm(pred[j] - pred[i])
                    - np.linalg.norm(g[j, :2] - g[i, :2]))

    along, across = np.array(along), np.array(across)
    print(f'\n{len(along)} keypoints with both edge endpoints labelled\n')
    print(f'{"component":<34}{"mean":>8}{"median":>9}{"p90":>8}')
    for lbl, v in [('ALONG the edge', along), ('ACROSS the edge', across)]:
        print(f'{lbl:<34}{v.mean():>8.2f}{np.median(v):>9.2f}'
              f'{np.percentile(v, 90):>8.2f}')
    print(f'\n  ratio of means : {along.mean() / across.mean():.2f}x'
          f'   (1.39x before the span term; 1.0 = isotropic)')

    print(f'\n{"keypoint":<12}{"along":>9}{"across":>9}{"ratio":>8}')
    for n in NAMES:
        a, c = np.array(per_kpt[n][0]), np.array(per_kpt[n][1])
        if len(a):
            print(f'{n:<12}{a.mean():>9.2f}{c.mean():>9.2f}'
                  f'{a.mean() / max(c.mean(), 1e-9):>8.2f}')

    print('\nsigned span error per flap  (predicted length - true length):')
    print(f'{"flap":<10}{"mean":>9}{"median":>9}{"% too SHORT":>13}')
    for name, _ in FLAPS:
        v = np.array(span_err[name])
        if len(v):
            print(f'{name:<10}{v.mean():>9.2f}{np.median(v):>9.2f}'
                  f'{(v < 0).mean() * 100:>12.0f}%')


if __name__ == '__main__':
    main()
