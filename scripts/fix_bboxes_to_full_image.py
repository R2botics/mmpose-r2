"""Rewrite COCO annotation bboxes to full-image [0, 0, W, H].

Whenever CVAT exports a new annotation JSON or synthbox generates a fresh
synthetic set, the bboxes come out tight around the object. Our training
pipeline consistently uses full-image bboxes (so the entire cropped image
becomes the topdown-affine input). This script standardises them.

Usage:
    # Fix a single file (backs up the original as .bak)
    python scripts/fix_bboxes_to_full_image.py path/to/person_keypoints_train.json

    # Fix multiple files in one call
    python scripts/fix_bboxes_to_full_image.py file1.json file2.json file3.json

    # Skip creating a .bak backup file
    python scripts/fix_bboxes_to_full_image.py --no-backup file.json

    # Dry run to see what would change without writing
    python scripts/fix_bboxes_to_full_image.py --dry-run file.json
"""
import argparse
import json
import shutil
from pathlib import Path


def is_full_image_bbox(bbox, w, h, tol=1.0):
    """Return True if `bbox` already covers the full image within `tol` px."""
    return (abs(bbox[0]) <= tol and
            abs(bbox[1]) <= tol and
            abs(bbox[2] - w) <= tol and
            abs(bbox[3] - h) <= tol)


def fix_file(path: Path, backup: bool, dry_run: bool) -> dict:
    """Rewrite bboxes in `path`. Returns a summary dict."""
    with open(path) as f:
        data = json.load(f)

    img_dims = {img['id']: (img['width'], img['height']) for img in data['images']}
    total = len(data['annotations'])
    already_full = 0
    fixed = 0

    for ann in data['annotations']:
        if ann['image_id'] not in img_dims:
            # Orphan annotation — skip
            continue
        w, h = img_dims[ann['image_id']]
        if is_full_image_bbox(ann['bbox'], w, h):
            already_full += 1
            continue
        ann['bbox'] = [0.0, 0.0, float(w), float(h)]
        ann['area'] = float(w * h)
        fixed += 1

    summary = {
        'path': str(path),
        'total': total,
        'already_full': already_full,
        'fixed': fixed,
        'would_fix': fixed,
        'wrote': False,
        'backup': 'n/a',
    }

    if dry_run or fixed == 0:
        return summary

    if backup:
        bak_path = path.with_suffix(path.suffix + '.bak')
        # Deliberately never overwrite an existing .bak: it is the pristine
        # original, and clobbering it on a second run would destroy the only
        # untouched copy. Report honestly which of the two happened.
        if not bak_path.exists():
            shutil.copy(path, bak_path)
            summary['backup'] = 'created'
        else:
            summary['backup'] = 'kept existing (NOT overwritten)'

    with open(path, 'w') as f:
        json.dump(data, f)

    summary['wrote'] = True
    return summary


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('files', nargs='+', type=Path,
                        help='One or more COCO annotation JSON files to fix')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create a .bak backup of the original')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would be changed without writing')
    args = parser.parse_args()

    for path in args.files:
        if not path.is_file():
            print(f'  SKIP  {path}  (not found)')
            continue

        summary = fix_file(
            path, backup=not args.no_backup, dry_run=args.dry_run)

        total = summary['total']
        already = summary['already_full']

        if args.dry_run:
            would_fix = summary['would_fix']
            print(f'  DRY   {path}')
            print(f'        total={total}  already_full={already}  would_fix={would_fix}')
        elif not summary['wrote']:
            print(f'  OK    {path}  (all {total} bboxes already full-image)')
        else:
            print(f'  FIXED {path}')
            print(f'        total={total}  already_full={already}  fixed={summary["fixed"]}')
            if not args.no_backup:
                print(f'        backup: {path.with_suffix(path.suffix + ".bak")}'
                      f'  [{summary.get("backup", "n/a")}]')


if __name__ == '__main__':
    main()
