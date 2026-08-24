"""Will this annotation set actually train? Check before launching a pretrain.

check_synth_coverage.py answers "is the GEOMETRY useful". This answers the
separate, blunter question: "will the dataloader survive epoch 1". A pretrain
that dies 300 iterations in on a missing depth file costs the same wall-clock
as one that dies at iteration 1.

CHECKS
  1. SCHEMA     8 keypoints, 24-long arrays, category names matching ours.
  2. CONVENTION the minimum-distance corner pairing (see
                diagnose_synth_pairs.py) -- renumbered indices would teach the
                wrong corner identities.
  3. FILES      color images resolve under <root>/images/ with the suffix
                LoadRGBDImage expects, and each has a matching depth file
                under <root>/depth/ with the same HxW. This is the check that
                actually catches broken sets.
  4. BBOXES     present and non-degenerate, and full-frame like every other
                set here (the images are pre-cropped to the box, so a TIGHT
                bbox is the anomaly and would shift the effective scale).
  5. BOUNDS     keypoints outside the image, which the RSC set historically had.

Usage:
    python scripts/preflight_pretrain_set.py <annotations.json> [--samples 200]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PAIRS = ((0, 7), (1, 2), (3, 4), (5, 6))
NAMES = ('NORTH_1', 'NORTH_2', 'EAST_1', 'EAST_2',
         'SOUTH_1', 'SOUTH_2', 'WEST_1', 'WEST_2')
COLOR_SUFFIX = '_cropped_color.png'
DEPTH_SUFFIX = '_aligned_depth.png'

_fail = []


def check(label, ok, detail='', fatal=True):
    tag = 'OK  ' if ok else ('FAIL' if fatal else 'WARN')
    print(f'  [{tag}] {label}{"   " + detail if detail else ""}')
    if not ok and fatal:
        _fail.append(label)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ann')
    p.add_argument('--samples', type=int, default=200,
                   help='how many images to stat on disk')
    args = p.parse_args()

    ann_path = Path(args.ann).expanduser()
    root = ann_path.parent.parent
    print(f'annotation : {ann_path}')
    print(f'data_root  : {root}   (inferred as the parent of annotations/)\n')

    data = json.loads(ann_path.read_text())
    imgs = {i['id']: i for i in data['images']}
    anns = data['annotations']
    print(f'  {len(imgs)} images, {len(anns)} annotations\n')

    # ---- 1. schema ---------------------------------------------------------
    print('1. SCHEMA')
    cats = data.get('categories') or []
    kp_names = list(cats[0].get('keypoints', [])) if cats else []
    check('category keypoint names match ours', kp_names == list(NAMES),
          '' if kp_names == list(NAMES) else f'file has {kp_names}')
    bad_len = sum(1 for a in anns if len(a.get('keypoints', [])) != 24)
    check('every annotation has 24 keypoint values', bad_len == 0,
          f'{bad_len} malformed' if bad_len else '')

    # ---- 2. convention -----------------------------------------------------
    print('\n2. CONVENTION')
    full = [np.array(a['keypoints'], float).reshape(-1, 3)[:, :2]
            for a in anns
            if len(a.get('keypoints', [])) == 24
            and (np.array(a['keypoints'], float).reshape(-1, 3)[:, 2] > 0).all()]
    if full:
        from itertools import product  # noqa: F401  (kept for clarity)

        def matchings(items):
            if not items:
                yield ()
                return
            first, rest = items[0], items[1:]
            for k in range(len(rest)):
                for tail in matchings(rest[:k] + rest[k + 1:]):
                    yield ((first, rest[k]),) + tail

        allm = list(matchings(tuple(range(8))))
        ours = set(map(frozenset, PAIRS))
        wins = 0
        for k in full:
            d = np.linalg.norm(k[:, None] - k[None, :], axis=-1)
            best = min(allm, key=lambda m: sum(d[i, j] for i, j in m))
            wins += set(map(frozenset, best)) == ours
        frac = wins / len(full)
        check('minimum-distance pairing recovers our corner convention',
              frac > 0.90, f'{frac:.1%} of {len(full)} fully-visible boxes')
    else:
        check('fully-visible boxes available to test convention', False,
              'none found', fatal=False)

    # ---- 3. files ----------------------------------------------------------
    print('\n3. FILES  (what LoadRGBDImage will do)')
    names = [i['file_name'] for i in imgs.values()]
    n_suffix = sum(1 for n in names if n.endswith(COLOR_SUFFIX))
    check(f'file_name ends with {COLOR_SUFFIX}', n_suffix == len(names),
          f'{n_suffix}/{len(names)}'
          + ('' if n_suffix == len(names)
             else f'   e.g. {Path(names[0]).name}'))

    rng = np.random.default_rng(0)
    sample = [names[i] for i in
              rng.choice(len(names), min(args.samples, len(names)), replace=False)]
    miss_c, miss_d, mismatch, dtypes = [], [], [], Counter()
    for n in sample:
        cpath = root / 'images' / n
        if not cpath.exists():
            miss_c.append(str(cpath))
            continue
        dpath = Path(str(cpath).replace('/images/', '/depth/')
                     .replace(COLOR_SUFFIX, DEPTH_SUFFIX))
        if not dpath.exists():
            miss_d.append(str(dpath))
            continue
        d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        c = cv2.imread(str(cpath))
        if d is None or c is None:
            mismatch.append(f'{n} unreadable')
        elif d.shape[:2] != c.shape[:2]:
            mismatch.append(f'{n} depth {d.shape[:2]} vs color {c.shape[:2]}')
        else:
            dtypes[str(d.dtype)] += 1
    check(f'color images resolve under {root}/images/', not miss_c,
          f'{len(miss_c)}/{len(sample)} missing, e.g. {miss_c[0]}' if miss_c else '')
    check(f'matching depth under {root}/depth/', not miss_d,
          f'{len(miss_d)}/{len(sample)} missing, e.g. {miss_d[0]}' if miss_d else '')
    check('depth and color have the same HxW', not mismatch,
          mismatch[0] if mismatch else '')
    if dtypes:
        check('depth dtype is uint16', set(dtypes) == {'uint16'},
              str(dict(dtypes)), fatal=False)

    # ---- 4. bboxes ---------------------------------------------------------
    print('\n4. BBOXES')
    whole, degen, tight = 0, 0, 0
    for a in anns:
        b = a.get('bbox')
        im = imgs.get(a['image_id'], {})
        if not b or b[2] <= 1 or b[3] <= 1:
            degen += 1
            continue
        if im.get('width') and b[2] >= im['width'] * 0.99 \
                and b[3] >= im['height'] * 0.99:
            whole += 1
        else:
            tight += 1
    check('no degenerate bboxes', degen == 0, f'{degen} with w or h <= 1')
    # Every set in this project -- RSC train, ChewyCrops, the old synthetic
    # set -- stores a WHOLE-FRAME bbox, because the images are already cropped
    # to the box (`_cropped_color`). So full-frame is CORRECT here, and a set
    # with TIGHT bboxes is the anomaly: it would crop tighter than everything
    # the model was trained on and change the effective scale.
    check('bboxes follow the project convention (whole frame)',
          whole >= 0.99 * max(len(anns), 1),
          f'{tight}/{len(anns)} are tight rather than full-frame -- every '
          f'other set in this project is full-frame', fatal=False)

    # ---- 5. bounds ---------------------------------------------------------
    print('\n5. BOUNDS')
    oob = 0
    for a in anns:
        if len(a.get('keypoints', [])) != 24:
            continue
        k = np.array(a['keypoints'], float).reshape(-1, 3)
        im = imgs.get(a['image_id'], {})
        w, h = im.get('width'), im.get('height')
        if not w:
            continue
        v = k[:, 2] > 0
        oob += int(((k[v, 0] < 0) | (k[v, 0] > w)
                    | (k[v, 1] < 0) | (k[v, 1] > h)).sum())
    check('labelled keypoints inside the image', oob == 0,
          f'{oob} out of bounds', fatal=False)

    print('\n' + ('  NO-GO: ' + '; '.join(_fail) if _fail
                  else '  GO -- the dataloader should survive this set'))


if __name__ == '__main__':
    main()
