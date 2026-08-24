"""Does a synthetic set cover the flap geometry that real boxes actually show?

Run this BEFORE spending a pretrain on a regenerated synthetic set. It is a
two-minute check against a multi-hour run.

WHY THIS METRIC
    The model collapses the two keypoints that mark one physical box corner
    (e.g. EAST_2 onto SOUTH_1). Measured on 428 real corner pairs with
    best_coco_AP_epoch_120, collapse rate against the OLD synthetic set's
    coverage:

        inside synthetic p90            402 pairs    7% collapsed
        synthetic p90-p99 (rare)          9 pairs   44% collapsed
        beyond synthetic p99 (unseen)    17 pairs   35% collapsed

    Collapse concentrates where synthetic coverage thins out -- and the WORST
    band is p90-p99, which the old set did cover, just rarely. So the failure
    is driven by DENSITY, not a hard coverage boundary: the model has seen
    those configurations and still treats them as outliers.

    Corner separation (normalised by box size) is the measurable proxy for how
    far the flaps are bent open, which is why it is what this script reports.

THE OLD SYNTHETIC SET, for reference (15045 boxes):

        median 0.066   p90 0.109   p99 0.139   CV 0.51

    versus real:

        RSC train      median 0.044   p90 0.135   p99 0.222   CV 0.94
        ChewyCrops     median 0.047   p90 0.106   p99 0.200   CV 0.72

    Synthetic sat at a HIGHER median with HALF the variability and a much
    shorter tail -- clustered around one "typical" flap opening while real
    boxes spread far wider.

WHAT GOOD LOOKS LIKE
    Not a higher maximum -- the old set already reached 1.06, above any real
    box. Measured against 1192 pooled real pairs, the old set was a narrow
    bump sitting too HIGH, with no tail:

        threshold   % real above   % old synth above
            0.080         16.4%             35.0%    <- synth 2x too many
            0.109         10.9%             10.0%    <- matched
            0.139          7.4%              1.0%    <- 7.6x too few
            0.180          4.7%              0.0%    <- essentially none
            0.220          0.8%              0.0%

        stat        real     old synth
        median     0.043       0.066       <- synth too HIGH
        p90        0.114       0.109
        p99        0.213       0.139       <- synth too LOW
        CV          0.89        0.51       <- synth too uniform

    So BOTH ends need to move: the median down and a long tail added. A set of
    uniformly wilder flap angles would fix the tail and make the median worse.

    The script prints PASS/WARN against the real distribution.

Usage:
    python scripts/check_synth_coverage.py <new_synth.json> [more.json ...]
    python scripts/check_synth_coverage.py <new.json> --compare-old
"""
import argparse
import json
from pathlib import Path

import numpy as np

# the four physical box corners; each is marked by an N/S keypoint and an E/W one
PAIRS = [(0, 7), (1, 2), (3, 4), (5, 6)]

REAL_SETS = [
    ('RSC train', 'data/RSC_Keypoints_RGBD/annotations/person_keypoints_train.json'),
    ('RSC val', 'data/RSC_Keypoints_RGBD/annotations/person_keypoints_val.json'),
    ('ChewyCrops', '/home/siddharth/Documents/ChewyCrops/annotations/person_keypoints_Train.json'),
    ('OOD', 'data/OOD/annotations/person_keypoints_Test.json'),
]
OLD_SYNTH = 'data/RSC_Keypoints_RGBD/annotations/person_keypoints_synth_only.json'


def separations(path):
    """Corner-pair separations, normalised by box size, for one COCO file."""
    with open(path) as f:
        data = json.load(f)
    out, n_boxes, n_partial = [], 0, 0
    for ann in data['annotations']:
        k = np.array(ann['keypoints'], float).reshape(-1, 3)
        if (k[:, 2] <= 0).any():
            n_partial += 1
            continue
        n_boxes += 1
        size = np.sqrt(np.ptp(k[:, 0]) * np.ptp(k[:, 1]))
        if size < 1e-6:
            continue
        for i, j in PAIRS:
            out.append(float(np.linalg.norm(k[j, :2] - k[i, :2])) / size)
    return np.array(out), n_boxes, n_partial


def describe(name, s, n_boxes):
    cv = s.std() / s.mean() if s.mean() else 0.0
    print(f'{name:<22}{n_boxes:>8}{len(s):>8}{np.median(s):>9.3f}'
          f'{np.percentile(s, 90):>8.3f}{np.percentile(s, 99):>8.3f}'
          f'{s.max():>8.3f}{cv:>7.2f}'
          f'{(s > 0.109).mean() * 100:>8.0f}%{(s > 0.139).mean() * 100:>8.0f}%')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('candidates', nargs='+', help='COCO JSON(s) for the new set')
    p.add_argument('--compare-old', action='store_true',
                   help='also show the previous synthetic set')
    args = p.parse_args()

    print('Corner-pair separation, normalised by box size')
    print('(proxy for how far the flaps are bent open)\n')
    print(f'{"set":<22}{"boxes":>8}{"pairs":>8}{"median":>9}{"p90":>8}'
          f'{"p99":>8}{"max":>8}{"CV":>7}{">.109":>9}{">.139":>9}')

    real = []
    for name, path in REAL_SETS:
        if not Path(path).exists():
            continue
        s, n, _ = separations(path)
        if len(s):
            describe(name, s, n)
            real.append(s)
    if args.compare_old and Path(OLD_SYNTH).exists():
        s, n, _ = separations(OLD_SYNTH)
        describe('SYNTH (old)', s, n)

    print()
    cand = None
    for c in args.candidates:
        s, n, partial = separations(c)
        if not len(s):
            print(f'{Path(c).name}: no fully-visible boxes')
            continue
        describe(f'CANDIDATE {Path(c).stem[:11]}', s, n)
        if partial:
            print(f'{"":22}({partial} boxes skipped: not all 8 keypoints visible)')
        cand = s if cand is None else np.concatenate([cand, s])

    if cand is None or not real:
        return
    allreal = np.concatenate(real)

    print('\nTARGETS (from the collapse analysis)')
    # Targets are the pooled real distribution +/- a tolerance, NOT numbers
    # anchored to the old synthetic set.
    checks = [
        ('median', float(np.median(cand)), 0.035, 0.055, ''),
        ('% pairs above 0.139', (cand > 0.139).mean() * 100, 5.0, 12.0, '%'),
        ('% pairs above 0.180', (cand > 0.180).mean() * 100, 3.0, 8.0, '%'),
        ('coefficient of variation', cand.std() / cand.mean(), 0.70, 1.05, ''),
    ]
    ok = True
    for label, value, lo, hi, unit in checks:
        good = lo <= value <= hi
        ok &= good
        print(f'  {label:<28}{value:>8.2f}{unit}   target {lo}-{hi}{unit}   '
              f'{"PASS" if good else "WARN"}')

    # how much of real geometry the candidate now supports
    hi99 = np.percentile(cand, 99)
    beyond = (allreal > hi99).mean() * 100
    print(f'\n  candidate p99 = {hi99:.3f}')
    print(f'  {beyond:.1f}% of REAL pairs still fall beyond it '
          f'(was 4.0% for the old set; lower is better)')
    print(f'\n  {"looks good — worth a pretrain" if ok and beyond < 4.0 else "distribution has NOT moved enough; regenerate before spending a pretrain"}')


if __name__ == '__main__':
    main()
