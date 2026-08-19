"""Fold an externally-annotated set into an existing COCO training split.

Handles the four things that silently corrupt a hand-merge:

  1. **ID collisions** — image_id / annotation id are renumbered past the base.
  2. **file_name scoping** — new images live under a subfolder of the data
     root, so file_name is rewritten to '<subdir>/<basename>'.
  3. **bbox convention** — CVAT exports tight bboxes; this project trains on
     full-image ones (see scripts/fix_bboxes_to_full_image.py). New entries are
     rewritten to [0, 0, W, H] to match.
  4. **category drift** — refuses to merge if the keypoint names or their ORDER
     differ between the two files, which would scramble the labels.

Partial annotations are preserved exactly: visibility flags (including v=0,
which CVAT writes with real coordinates rather than zeros) pass through
untouched, and num_keypoints is recomputed from them.

The images themselves are exposed with two directory symlinks:
    <base_root>/images/<subdir> -> <new_dir>/images
    <base_root>/depth/<subdir>  -> <new_dir>/depth
which keeps LoadRGBDImage's '/images/' -> '/depth/' path rewrite working.

Usage:
    python scripts/add_dataset.py \\
        --base-root data/RSC_Keypoints_RGBD \\
        --base-ann  annotations/person_keypoints_train.json \\
        --new-dir   /home/siddharth/Documents/ChewyCrops2 \\
        --new-ann   /home/siddharth/Documents/ChewyCrops2/annotations/person_keypoints_Train.json \\
        --subdir    chewy2 \\
        --out-ann   annotations/person_keypoints_train_plus_chewy2.json \\
        --link
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument('--base-root', required=True,
                    help='Existing data root, e.g. data/RSC_Keypoints_RGBD')
    ap.add_argument('--base-ann', required=True,
                    help='Annotation file to extend, relative to --base-root')
    ap.add_argument('--new-dir', required=True,
                    help='Folder holding the new images/ and depth/')
    ap.add_argument('--new-ann', required=True,
                    help='COCO json for the new set')
    ap.add_argument('--subdir', required=True,
                    help='Subfolder name to expose the new set under')
    ap.add_argument('--out-ann', required=True,
                    help='Output annotation file, relative to --base-root')
    ap.add_argument('--link', action='store_true',
                    help='Actually create the two directory symlinks')
    ap.add_argument('--keep-bbox', action='store_true',
                    help='Keep the incoming tight bboxes instead of '
                         'rewriting them to full-image (not recommended)')
    args = ap.parse_args()

    base_root = Path(args.base_root)
    new_dir = Path(args.new_dir).resolve()
    base = load(base_root / args.base_ann)
    new = load(args.new_ann)

    # ---- 4. category drift ------------------------------------------------
    bk = base['categories'][0]['keypoints']
    nk = new['categories'][0]['keypoints']
    if bk != nk:
        raise SystemExit(
            'Keypoint names/order differ between the two annotation files.\n'
            f'  base: {bk}\n  new : {nk}\n'
            'Merging these would scramble the labels. Aborting.')
    print(f'categories match ({len(bk)} keypoints, same order)')

    # ---- verify the new files exist, with depth partners -------------------
    new_imgs = {im['id']: im for im in new['images']}
    missing_c, missing_d = [], []
    for im in new['images']:
        stem = os.path.basename(im['file_name'])
        if not (new_dir / 'images' / stem).exists():
            missing_c.append(stem)
        d = stem.replace('_cropped_color.png', '_aligned_depth.png')
        if not (new_dir / 'depth' / d).exists():
            missing_d.append(d)
    if missing_c or missing_d:
        raise SystemExit(
            f'{len(missing_c)} colour and {len(missing_d)} depth files named in '
            f'{args.new_ann} are absent from {new_dir}. First missing: '
            f'{(missing_c or missing_d)[:3]}')
    print(f'all {len(new["images"])} colour + depth pairs present on disk')

    # ---- 1/2/3. renumber, rescope, rewrite bbox ---------------------------
    id0 = max([im['id'] for im in base['images']] + [0])
    ann0 = max([a['id'] for a in base['annotations']] + [0])
    old2new = {}
    add_imgs, add_anns = [], []
    for n, im in enumerate(new['images'], start=1):
        nid = id0 + n
        old2new[im['id']] = nid
        rec = dict(im)
        rec['id'] = nid
        rec['file_name'] = f'{args.subdir}/{os.path.basename(im["file_name"])}'
        add_imgs.append(rec)
    for n, a in enumerate(new['annotations'], start=1):
        rec = dict(a)
        rec['id'] = ann0 + n
        rec['image_id'] = old2new[a['image_id']]
        im = new_imgs[a['image_id']]
        if not args.keep_bbox:
            rec['bbox'] = [0, 0, im['width'], im['height']]
            rec['area'] = float(im['width'] * im['height'])
        # recompute from the visibility flags actually present
        kp = rec['keypoints']
        rec['num_keypoints'] = sum(1 for i in range(2, len(kp), 3) if kp[i] > 0)
        add_anns.append(rec)

    merged = dict(base)
    merged['images'] = base['images'] + add_imgs
    merged['annotations'] = base['annotations'] + add_anns

    # ---- sanity ------------------------------------------------------------
    ids = [i['id'] for i in merged['images']]
    aids = [a['id'] for a in merged['annotations']]
    fns = [i['file_name'] for i in merged['images']]
    assert len(ids) == len(set(ids)), 'duplicate image ids after merge'
    assert len(aids) == len(set(aids)), 'duplicate annotation ids after merge'
    assert len(fns) == len(set(fns)), 'duplicate file_names after merge'
    known = {i['id'] for i in merged['images']}
    assert all(a['image_id'] in known for a in merged['annotations']), \
        'annotation references a missing image_id'

    out = base_root / args.out_ann
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(merged, f, indent=1)

    # ---- symlinks ----------------------------------------------------------
    links = [(base_root / 'images' / args.subdir, new_dir / 'images'),
             (base_root / 'depth' / args.subdir, new_dir / 'depth')]
    if args.link:
        for dst, src in links:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
            print(f'linked {dst} -> {src}')
    else:
        print('\nsymlinks NOT created (pass --link). Equivalent commands:')
        for dst, src in links:
            print(f'  ln -sfn {src} {dst}')

    nkc = Counter(a['num_keypoints'] for a in add_anns)
    print(f'\nbase   : {len(base["images"]):4d} images, {len(base["annotations"]):4d} annotations')
    print(f'added  : {len(add_imgs):4d} images, {len(add_anns):4d} annotations '
          f'(image ids {id0+1}..{id0+len(add_imgs)})')
    print(f'merged : {len(merged["images"]):4d} images, {len(merged["annotations"]):4d} annotations')
    print(f'added num_keypoints spread: {dict(sorted(nkc.items()))}')
    if not args.keep_bbox:
        print('added bboxes rewritten to full-image [0,0,W,H]')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
