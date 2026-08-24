"""Hold out part of a synthetic set so a pretrain can tell converged from overfit.

A pretrain validated only on REAL data cannot distinguish "still learning" from
"has started memorising synthetic": both show real-set metrics going flat or
backwards. A held-out synthetic split separates them. If synthetic val keeps
improving while real val degrades, the run is overfitting the generator and
should stop; if both flatten, it has converged.

The split is by IMAGE and deterministic -- membership comes from a hash of the
file name, so it is stable across reruns, independent of annotation order, and
identical on any machine. Rerunning never reshuffles a box from val into train,
which would quietly leak the holdout.

The holdout must NEVER drive checkpoint selection. Give it `select=False` in
the multi-domain metric so it is reported and ignored:

    dict(name='synth_new', data_root=..., select=False)

Usage:
    python scripts/make_synth_val_split.py <annotations.json> [--n 400]
    python scripts/make_synth_val_split.py <a.json> <b.json> --frac 0.03
"""
import argparse
import hashlib
import json
from pathlib import Path


def split_key(file_name: str) -> int:
    """Stable per-image hash. Not Python's hash(): that is salted per process."""
    return int(hashlib.md5(file_name.encode('utf-8')).hexdigest()[:8], 16)


def split_file(path: Path, n_val: int, frac: float, dry_run: bool) -> dict:
    data = json.loads(path.read_text())
    images = data['images']

    n = n_val if n_val else max(1, round(len(images) * frac))
    n = min(n, len(images) - 1)          # never hold out everything

    ranked = sorted(images, key=lambda im: split_key(im['file_name']))
    val_ids = {im['id'] for im in ranked[:n]}

    def subset(keep_val: bool):
        ids = {im['id'] for im in images if (im['id'] in val_ids) == keep_val}
        out = {k: v for k, v in data.items() if k not in ('images', 'annotations')}
        out['images'] = [im for im in images if im['id'] in ids]
        out['annotations'] = [a for a in data['annotations']
                              if a['image_id'] in ids]
        return out

    train, val = subset(False), subset(True)

    # A leak here would silently invalidate the whole point of the holdout.
    tn = {im['file_name'] for im in train['images']}
    vn = {im['file_name'] for im in val['images']}
    assert not (tn & vn), f'{len(tn & vn)} file names in BOTH splits'
    assert len(tn) + len(vn) == len(images), 'images lost in the split'

    paths = {'train': path.with_name(path.stem + '_trainsplit.json'),
             'val': path.with_name(path.stem + '_valsplit.json')}
    if not dry_run:
        paths['train'].write_text(json.dumps(train))
        paths['val'].write_text(json.dumps(val))

    return dict(src=path.name, total=len(images),
                n_train=len(train['images']), n_val=len(val['images']),
                ann_train=len(train['annotations']),
                ann_val=len(val['annotations']), paths=paths)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+', type=Path)
    p.add_argument('--n', type=int, default=0,
                   help='hold out exactly this many images (overrides --frac)')
    p.add_argument('--frac', type=float, default=0.03,
                   help='hold out this fraction (default 3%%)')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    for f in args.files:
        if not f.is_file():
            print(f'  SKIP  {f}  (not found)')
            continue
        s = split_file(f, args.n, args.frac, args.dry_run)
        print(f'  {"DRY  " if args.dry_run else "SPLIT"} {s["src"]}'
              f'   {s["total"]} imgs -> train {s["n_train"]} '
              f'({s["ann_train"]} anns) / val {s["n_val"]} '
              f'({s["ann_val"]} anns)')
        if not args.dry_run:
            for k, v in s['paths'].items():
                print(f'          {k:<6} {v}')


if __name__ == '__main__':
    main()
