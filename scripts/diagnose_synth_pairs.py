"""Is a synthetic set's keypoint CONVENTION the same as the real sets'?

Run this when check_synth_coverage.py reports a candidate whose separations are
wildly larger than real -- BEFORE concluding the flaps are merely splayed too
far. A convention mismatch produces the same symptom and is far more damaging:
pretraining on scrambled indices teaches the wrong corner identities.

THREE INDEPENDENT CHECKS

  1. PAIRING. Each physical box corner is marked by two keypoints, so the
     correct pairing is the one that minimises total within-pair distance.
     Brute-force all 105 perfect matchings of 8 points into 4 pairs, per box,
     and report the modal winner. On a set using our convention this recovers
     ((0,7),(1,2),(3,4),(5,6)). Anything else is a mismatch.

  2. NORMALISATION. check_synth_coverage divides by sqrt(ptp_x * ptp_y), the
     geometric mean of the keypoint extent. That collapses toward zero for a
     near-degenerate (very elongated) layout and inflates the ratio without the
     flaps being open at all. Reported here as the aspect distribution plus raw
     pixel separations, which are convention-independent.

  3. LAYOUT. Mean position of each keypoint index in a box-normalised frame.
     Our convention puts NORTH at low y, SOUTH at high y, and so on; a rotated
     or renumbered set shows up immediately.

Always run a real set first as a CONTROL -- if check 1 does not recover our
pairing on RSC train, the method is broken, not the candidate.

Usage:
    python scripts/diagnose_synth_pairs.py <file.json> [more.json ...]
"""
import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

PAIRS = ((0, 7), (1, 2), (3, 4), (5, 6))
NAMES = ('NORTH_1', 'NORTH_2', 'EAST_1', 'EAST_2',
         'SOUTH_1', 'SOUTH_2', 'WEST_1', 'WEST_2')


def all_matchings(items):
    """Every way to split a tuple of 8 indices into 4 unordered pairs (105)."""
    if not items:
        yield ()
        return
    first, rest = items[0], items[1:]
    for k in range(len(rest)):
        pair = (first, rest[k])
        for tail in all_matchings(rest[:k] + rest[k + 1:]):
            yield (pair,) + tail


MATCHINGS = list(all_matchings(tuple(range(8))))


def load(path):
    """Fully-visible 8-keypoint arrays from one COCO file."""
    with open(path) as f:
        data = json.load(f)
    boxes = []
    for ann in data['annotations']:
        k = np.array(ann['keypoints'], float).reshape(-1, 3)
        if len(k) != 8 or (k[:, 2] <= 0).any():
            continue
        boxes.append(k[:, :2])
    return boxes, data.get('categories', [])


def report(name, path):
    boxes, cats = load(path)
    print(f'\n{"=" * 72}\n{name}   ({len(boxes)} fully-visible boxes)\n{"=" * 72}')
    if not boxes:
        print('  no usable boxes')
        return

    # --- declared keypoint names, if the file carries them -------------------
    kp_names = cats[0].get('keypoints') if cats else None
    if kp_names:
        same = list(kp_names) == list(NAMES)
        print(f'  declared names : {"MATCH" if same else "DIFFER"}')
        if not same:
            print(f'    file : {list(kp_names)}')
            print(f'    ours : {list(NAMES)}')
    else:
        print('  declared names : (none in file)')

    # --- 1. which pairing actually marks the physical corners ----------------
    votes, assumed_rank = Counter(), []
    for k in boxes:
        d = np.linalg.norm(k[:, None, :] - k[None, :, :], axis=-1)
        totals = np.array([sum(d[i, j] for i, j in m) for m in MATCHINGS])
        votes[MATCHINGS[int(totals.argmin())]] += 1
        ours = sum(d[i, j] for i, j in PAIRS)
        assumed_rank.append(int((totals < ours - 1e-9).sum()) + 1)

    print('\n  1. PAIRING  (minimum-total-distance matching, per box)')
    for m, n in votes.most_common(3):
        tag = '  <-- our convention' if set(map(frozenset, m)) == set(
            map(frozenset, PAIRS)) else ''
        print(f'     {n:>6} boxes ({n / len(boxes):>5.1%})  '
              f'{tuple(tuple(p) for p in m)}{tag}')
    rank = np.array(assumed_rank)
    print(f'     our pairing ranks #1 in {(rank == 1).mean():>5.1%} of boxes '
          f'(median rank {int(np.median(rank))} of {len(MATCHINGS)})')

    # --- 2. normalisation sanity --------------------------------------------
    ptp = np.array([[np.ptp(k[:, 0]), np.ptp(k[:, 1])] for k in boxes])
    aspect = ptp.max(1) / np.maximum(ptp.min(1), 1e-6)
    size = np.sqrt(ptp[:, 0] * ptp[:, 1])
    raw = np.array([[np.linalg.norm(k[j] - k[i]) for i, j in PAIRS]
                    for k in boxes])
    print('\n  2. NORMALISATION')
    print(f'     keypoint extent px : x {np.median(ptp[:, 0]):>7.1f}   '
          f'y {np.median(ptp[:, 1]):>7.1f}   (medians)')
    print(f'     aspect ratio       : median {np.median(aspect):>5.2f}   '
          f'p99 {np.percentile(aspect, 99):>6.2f}   '
          f'max {aspect.max():>7.2f}')
    print(f'     degenerate (>5:1)  : {(aspect > 5).mean():>6.1%} of boxes')
    print(f'     RAW pair sep px    : median {np.median(raw):>7.2f}   '
          f'p99 {np.percentile(raw, 99):>8.2f}')
    print(f'     box size px        : median {np.median(size):>7.1f}')

    # --- 3. layout in a box-normalised frame ---------------------------------
    print('\n  3. LAYOUT  (mean position, 0-1 within the keypoint extent)')
    norm = np.array([(k - k.min(0)) / np.maximum(np.ptp(k, 0), 1e-6)
                     for k in boxes])
    mean = norm.mean(0)
    for i, nm in enumerate(NAMES):
        print(f'     [{i}] {nm:<9} x {mean[i, 0]:>5.2f}   y {mean[i, 1]:>5.2f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+')
    args = p.parse_args()
    for f in args.files:
        if Path(f).exists():
            report(Path(f).stem, f)
        else:
            print(f'\nmissing: {f}')


if __name__ == '__main__':
    main()
