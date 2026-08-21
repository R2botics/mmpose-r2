"""Do two models fail on the SAME keypoints, or on different ones?

Reads the per_keypoint_errors.csv written by compare_rgb_vs_rgbd.py (which must
have been run with two or more --model triples and an --ann-file) and answers
three questions:

  1. How correlated are the two models' errors? Highly correlated models share a
     failure mode, so ensembling them buys nothing.
  2. What would an ORACLE gain -- one that picks the better model per keypoint?
     That is the ceiling for any ensemble or agreement-based selector.
  3. Can DISAGREEMENT between the models flag the bad predictions WITHOUT ground
     truth? This is the part that works at inference time.

Usage:
    python scripts/model_agreement.py path/to/per_keypoint_errors.csv
    python scripts/model_agreement.py <csv> --bad-px 20 --by-camera
"""
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--bad-px', type=float, default=20.0,
                    help='Error above which a keypoint counts as unusable')
    ap.add_argument('--models', nargs=2, default=None,
                    help='Which two model labels to compare (default: the '
                         'first two found in the file)')
    ap.add_argument('--by-camera', action='store_true',
                    help='Also split by camera serial parsed from the filename')
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    labels = args.models or list(dict.fromkeys(d.model))[:2]
    if len(labels) < 2:
        raise SystemExit(f'need two models in {args.csv}, found {labels}')
    a, b = labels

    p = d[d.model.isin(labels)].pivot_table(
        'error_px', ['image', 'keypoint'], 'model').dropna()
    if a not in p or b not in p:
        raise SystemExit(f'missing {a!r} or {b!r} in {sorted(set(d.model))}')

    bad = args.bad_px
    print(f'{len(p)} keypoints scored by both\n')
    print(f'{"":<34}{"mean":>8}{"median":>9}{f">{bad:.0f}px":>8}{"max":>8}')
    rows = [(a, p[a]), (b, p[b]),
            ('mean of the two', p.mean(axis=1)),
            ('ORACLE: pick the better model', p.min(axis=1))]
    for lbl, v in rows:
        print(f'{lbl:<34}{v.mean():>8.2f}{v.median():>9.2f}'
              f'{int((v > bad).sum()):>8d}{v.max():>8.1f}')

    ba, bb = p[a] > bad, p[b] > bad
    print(f'\n  both models bad : {int((ba & bb).sum()):>3}   <- irreducible')
    print(f'  exactly one bad : {int((ba ^ bb).sum()):>3}   <- an ensemble could recover')
    print(f'  pearson r  : {p[a].corr(p[b]):.3f}')
    print(f'  spearman r : {p[a].corr(p[b], method="spearman"):.3f}')
    print('  (r near 1 means a shared failure mode -- ensembling will not help.'
          '\n   r well below 1 means independent failures -- it will.)')

    # The inference-time question: no ground truth, only the two predictions.
    print('\n  Can |error_a - error_b| flag the bad ones without ground truth?')
    dis = (p[a] - p[b]).abs()
    anybad = ba | bb
    for t in (5, 10, 20):
        f = dis > t
        if not f.sum():
            continue
        caught = int((f & anybad).sum())
        print(f'    |diff|>{t:>2}px flags {int(f.sum()):>3} kpts, catches '
              f'{caught:>2}/{int(anybad.sum())} bad '
              f'({caught / max(int(anybad.sum()), 1) * 100:.0f}%), '
              f'{int((f & ~anybad).sum()):>3} false alarms')

    if args.by_camera:
        q = p.reset_index()
        q['cam'] = q.image.str.extract(r'_(CPB[A-Z0-9]+)_')
        if q.cam.notna().any():
            print('\n  by camera:')
            g = q.groupby('cam')[[a, b]].agg(['mean', 'median'])
            print(g.round(2).to_string())


if __name__ == '__main__':
    main()
