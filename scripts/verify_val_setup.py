"""Check a config's validation setup BEFORE committing a training run to it.

The multi-domain metrics only warn about an empty domain at the first
validation, which on a 100-epoch pretrain is ten epochs in. This answers the
same questions in seconds, without a GPU or a checkpoint.

WHAT IT CHECKS
  0. REGISTRY every evaluator and val transform actually BUILDS through the
              registry, with the config's own custom_imports applied. Importing
              a metric class directly proves nothing: mmengine resolves it by
              name, and a module missing from custom_imports fails at
              build_val_loop -- after the model is built and an MLflow run has
              been opened.
  1. FILES    every val ann_file exists and parses; a sample of color and
              depth images resolve where LoadRGBDImage will look for them.
  2. ROUTING  each val image is pushed through the metric's OWN _route(), so
              a domain that silently receives nothing -- or steals another
              domain's samples -- shows up here rather than at epoch 10. This
              is the check that matters when two domains share a data_root.
  3. LEAKAGE  no file name appears in both a train and a val subset. A leak
              makes every validation number optimistic and is invisible at
              runtime.
  4. POWER    per-domain pair counts, and how many sit in the WIDE regime
              (>0.139 normalised corner separation). A domain with almost no
              wide pairs cannot detect a change that only affects wide boxes,
              so a null result there means "could not see it", not "no effect".

Usage:
    python scripts/verify_val_setup.py configs/<config>.py [--samples 40]
"""
import argparse
import copy
import json
import os.path as osp
import sys
from collections import Counter
from pathlib import Path

# Run as a script, sys.path[0] is scripts/ and the repo root is absent, so
# `mmpose::` in a _base_ would resolve to the installed package instead of
# this checkout -- and the installed one has no .mim/model-index.yml.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                            # noqa: E402
from mmengine.config import Config                            # noqa: E402

PAIRS = ((0, 7), (1, 2), (3, 4), (5, 6))
COLOR_SUFFIX = '_cropped_color.png'
DEPTH_SUFFIX = '_aligned_depth.png'

_fail = []


def check(label, ok, detail='', fatal=True):
    tag = 'OK  ' if ok else ('FAIL' if fatal else 'WARN')
    print(f'  [{tag}] {label}{"   " + detail if detail else ""}')
    if not ok and fatal:
        _fail.append(label)
    return ok


def subsets(dl):
    """The CocoDataset entries behind a dataloader, combined or not."""
    ds = dl['dataset']
    return list(ds['datasets']) if ds.get('type') == 'CombinedDataset' else [ds]


def img_paths(sub):
    root = sub['data_root']
    prefix = (sub.get('data_prefix') or {}).get('img', '')
    ann = osp.join(root, sub['ann_file'])
    if not osp.exists(ann):
        return ann, None, None
    data = json.loads(open(ann).read())
    paths = [osp.join(root, prefix, im['file_name']) for im in data['images']]
    return ann, data, paths


def separations(data):
    out = []
    for a in data['annotations']:
        k = np.array(a['keypoints'], float).reshape(-1, 3)
        if (k[:, 2] <= 0).any():
            continue
        size = np.sqrt(np.ptp(k[:, 0]) * np.ptp(k[:, 1]))
        if size < 1e-6:
            continue
        for i, j in PAIRS:
            out.append(np.linalg.norm(k[j, :2] - k[i, :2]) / size)
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--samples', type=int, default=40,
                   help='images per subset to stat on disk')
    args = p.parse_args()

    cfg = Config.fromfile(args.config)

    # ---- 0. registry ------------------------------------------------------
    print('0. REGISTRY')
    from mmengine.registry import init_default_scope
    from mmengine.utils import import_modules_from_strings
    from mmpose.registry import METRICS, TRANSFORMS

    init_default_scope(cfg.get('default_scope', 'mmpose'))
    ci = cfg.get('custom_imports')
    if ci:
        import_modules_from_strings(**ci)
    check('config declares custom_imports', bool(ci),
          '' if ci else
          'none declared -- custom metrics/transforms will not resolve',
          fatal=False)

    _evs = cfg.val_evaluator if isinstance(cfg.val_evaluator, list) \
        else [cfg.val_evaluator]
    for e in _evs:
        name = e['type']
        try:
            METRICS.build(copy.deepcopy(e))
            check(f'{name} builds', True)
        except KeyError as err:
            check(f'{name} builds', False,
                  f'not registered -- add its module to custom_imports '
                  f'({str(err)[:60]})')
        except Exception as err:                       # noqa: BLE001
            # Registered but unhappy (e.g. a missing ann_file). Still worth
            # reporting, but it is a different failure from a missing import.
            check(f'{name} builds', False,
                  f'{type(err).__name__}: {str(err)[:80]}', fatal=False)

    _pipe = cfg.val_dataloader['dataset'].get('pipeline') or []
    unreg = [s['type'] for s in _pipe if s['type'] not in TRANSFORMS]
    check('every val transform is registered', not unreg, f'missing: {unreg}')

    val_subs = subsets(cfg.val_dataloader)
    train_subs = subsets(cfg.train_dataloader)
    print(f'config : {args.config}')
    print(f'         {len(train_subs)} train subset(s), '
          f'{len(val_subs)} val subset(s)\n')

    # ---- 1. files ---------------------------------------------------------
    print('1. FILES')
    loaded, all_val_paths = {}, []
    rng = np.random.default_rng(0)
    for sub in val_subs:
        ann, data, paths = img_paths(sub)
        if data is None:
            check(f'{sub["ann_file"]}', False, f'missing: {ann}')
            continue
        loaded[ann] = (sub, data, paths)
        all_val_paths += paths
        miss = []
        idx = rng.choice(len(paths), min(args.samples, len(paths)), replace=False)
        for i in idx:
            c = paths[i]
            d = c.replace('/images/', '/depth/').replace(
                COLOR_SUFFIX, DEPTH_SUFFIX)
            if not osp.exists(c):
                miss.append(c)
            elif not osp.exists(d):
                miss.append(d)
        check(f'{osp.basename(ann)}', not miss,
              f'{len(data["images"])} imgs, {len(data["annotations"])} anns'
              + (f'; {len(miss)}/{len(idx)} sampled files missing, '
                 f'e.g. {miss[0]}' if miss else ''),
              fatal=bool(miss))

    # ---- 2. routing -------------------------------------------------------
    print('\n2. ROUTING  (through the metric that will score them)')
    evs = cfg.val_evaluator if isinstance(cfg.val_evaluator, list) \
        else [cfg.val_evaluator]
    dom_ev = next((e for e in evs if 'domains' in e), None)
    if dom_ev is None:
        check('multi-domain evaluator present', False,
              'val_evaluator has no `domains`; per-domain routing not used',
              fatal=False)
    else:
        from mmpose.evaluation.metrics.multi_domain_distance_metric import \
            MultiDomainKeypointDistanceMetric as M
        m = M(domains=dom_ev['domains'])
        hist, unrouted = Counter(), []
        for path in all_val_paths:
            try:
                hist[m.domain_names[m._route(path)]] += 1
            except ValueError:
                unrouted.append(path)
        check('every val image routes to a domain', not unrouted,
              f'{len(unrouted)} unrouted, e.g. {unrouted[0]}' if unrouted else '')
        empty = [n for n in m.domain_names if not hist[n]]
        check('every configured domain receives samples', not empty,
              f'empty: {empty}')
        sel = dict(zip(m.domain_names, m.domain_select))
        for n in m.domain_names:
            role = 'selects' if sel[n] else 'reported only'
            print(f'         {n:<12}{hist[n]:>6} imgs   ({role})')

    # ---- 3. leakage -------------------------------------------------------
    print('\n3. LEAKAGE  (train vs val file names)')
    train_names = set()
    for sub in train_subs:
        ann, data, _ = img_paths(sub)
        if data is None:
            check(f'train {sub["ann_file"]}', False, f'missing: {ann}')
            continue
        train_names |= {im['file_name'] for im in data['images']}
    for ann, (sub, data, _) in loaded.items():
        names = {im['file_name'] for im in data['images']}
        overlap = names & train_names
        check(f'{osp.basename(ann)} disjoint from train', not overlap,
              f'{len(overlap)} shared file names, e.g. '
              f'{sorted(overlap)[0]}' if overlap else f'{len(names)} names')

    # ---- 4. power ---------------------------------------------------------
    print('\n4. POWER  (can each domain see a wide-separation change?)')
    print(f'    {"subset":<46}{"pairs":>7}{">0.139":>10}')
    for ann, (sub, data, _) in loaded.items():
        s = separations(data)
        if not len(s):
            continue
        wide = int((s > 0.139).sum())
        print(f'    {osp.basename(ann)[:45]:<46}{len(s):>7}'
              f'{wide:>6} ({wide / len(s) * 100:>2.0f}%)')

    print('\n' + ('  NOT READY: ' + '; '.join(_fail) if _fail
                  else '  READY -- validation setup is consistent'))


if __name__ == '__main__':
    main()
