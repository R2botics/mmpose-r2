"""Run a trained pose model over a folder of images and emit COCO keypoints
JSON ready to import into CVAT for human correction (pre-labelling).

Folder structure expected (same as compare_rgb_vs_rgbd.py):
    <input_dir>/
        images/<stem>_cropped_color.png
        depth/<stem>_aligned_depth.png     (only needed for RGBD models)

Outputs:
    <output>.json        COCO keypoints, CVAT-import ready
    <output>_review.csv  per-image triage list, most-suspicious first

Why the review CSV: the model's confidence score does NOT catch its worst
failure mode. Corner-identity confusion — where a junction pair such as
EAST_2+SOUTH_1 is placed at the wrong corner of the box — has been observed at
score >0.99. What *does* catch it is a geometric consistency check: on a
rectangular box the flap edges must be pairwise perpendicular/parallel, and a
mis-assigned corner breaks that badly. This script scores every prediction on
both axes so the reviewer can start where the model is most likely wrong.

Usage:
    python scripts/predict_to_coco.py \\
        --input-dir /home/siddharth/Documents/ChewyCrops \\
        --config configs/hrnet-w32_8-kp_udp_rgbd_geom_finetune_real.py \\
        --checkpoint work_dirs/hrnet_synth_then_real_test_02/best_coco_AP_epoch_100.pth \\
        --output work_dirs/chewy_prelabel/annotations.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Reuse the *exact* model-loading and inference path used by the scoring
# harness, so pre-labels and evaluation cannot silently diverge.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_rgb_vs_rgbd import load_model, run_inference  # noqa: E402

from mmengine.registry import init_default_scope  # noqa: E402

KEYPOINT_NAMES = ['NORTH_1', 'NORTH_2', 'EAST_1', 'EAST_2',
                  'SOUTH_1', 'SOUTH_2', 'WEST_1', 'WEST_2']
# 1-indexed, matching the CVAT export this project already uses
SKELETON = [[3, 4], [1, 2], [5, 6], [7, 8]]

# Flap edges as (start, end) keypoint indices, 0-indexed
EDGES = dict(NORTH=(0, 1), EAST=(2, 3), SOUTH=(4, 5), WEST=(6, 7))
PERP_PAIRS = [('NORTH', 'EAST'), ('NORTH', 'WEST'),
              ('SOUTH', 'EAST'), ('SOUTH', 'WEST')]
PARALLEL_PAIRS = [('NORTH', 'SOUTH'), ('EAST', 'WEST')]


def geometry_violations(kpts, tolerance_deg=10.0):
    """Angle violations against the box's rectangularity priors, in degrees.

    Mirrors the constraints GeometricKeypointMSELoss trains on, but applied at
    inference as a validator. Returns (max_perp_violation, max_par_violation);
    0.0 means fully consistent within tolerance.
    """
    def unit(name):
        a, b = EDGES[name]
        v = kpts[b] - kpts[a]
        n = np.linalg.norm(v)
        return None if n < 1e-6 else v / n

    def angle_between(u, v):
        # in [0, 90]: undirected angle, so a flipped edge is not penalised
        return float(np.degrees(np.arccos(np.clip(abs(float(u @ v)), 0.0, 1.0))))

    perp, par = 0.0, 0.0
    for a, b in PERP_PAIRS:
        u, v = unit(a), unit(b)
        if u is None or v is None:
            continue
        perp = max(perp, max(0.0, (90.0 - angle_between(u, v)) - tolerance_deg))
    for a, b in PARALLEL_PAIRS:
        u, v = unit(a), unit(b)
        if u is None or v is None:
            continue
        par = max(par, max(0.0, angle_between(u, v) - tolerance_deg))
    return perp, par


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument('--input-dir', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True, help='Path to the COCO .json')
    p.add_argument('--existing', default=None,
                   help='COCO json of images a human has ALREADY annotated. '
                        'Those images are skipped so human labels are never '
                        'overwritten by model predictions.')
    p.add_argument('--merge-existing', action='store_true',
                   help='Also copy the --existing annotations into the output, '
                        'producing a single file to import into CVAT.')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--tolerance-deg', type=float, default=10.0,
                   help='Angular free-zone before a geometry violation is '
                        'counted (default 10, matching the training loss)')
    p.add_argument('--flag-score', type=float, default=0.5,
                   help='Keypoints below this score are flagged for review')
    p.add_argument('--mark-low-score-occluded', action='store_true',
                   help='Write visibility=1 instead of 2 for keypoints below '
                        '--flag-score, so CVAT shows them as occluded')
    args = p.parse_args()

    init_default_scope('mmpose')

    in_dir = Path(args.input_dir)
    color_paths = sorted((in_dir / 'images').glob('*_cropped_color.png'))
    if not color_paths:
        raise SystemExit(f'No *_cropped_color.png found under {in_dir}/images')
    print(f'Found {len(color_paths)} images in {in_dir}')

    existing_doc, already = None, set()
    if args.existing:
        existing_doc = json.load(open(args.existing))
        already = {im['file_name'] for im in existing_doc['images']}
        color_paths = [c for c in color_paths if c.name not in already]
        print(f'  {len(already)} already annotated by a human -> skipped')
        print(f'  {len(color_paths)} left to pre-label')
        if not color_paths:
            raise SystemExit('Nothing left to pre-label.')

    print(f'Loading model\n  config:     {args.config}\n'
          f'  checkpoint: {args.checkpoint}')
    model, pipeline, needs_depth, cfg = load_model(
        args.config, args.checkpoint, args.device)
    print(f'  input channels: {len(cfg.model.data_preprocessor.mean)} '
          f'(RGBD={needs_depth})')

    images, annotations, review = [], [], []
    # Start ids after the existing ones so a merged file has no collisions.
    id0 = 0
    if existing_doc and args.merge_existing:
        id0 = max([im['id'] for im in existing_doc['images']] +
                  [a['id'] for a in existing_doc['annotations']] + [0])
    for n, cp in enumerate(color_paths, start=1):
        i = id0 + n
        kpts, scores, color = run_inference(model, pipeline, cp)
        h, w = color.shape[:2]

        flat, n_labelled = [], 0
        for (x, y), s in zip(kpts, scores):
            v = 1 if (args.mark_low_score_occluded and s < args.flag_score) else 2
            flat += [round(float(x), 2), round(float(y), 2), v]
            n_labelled += 1

        images.append(dict(
            id=i, width=int(w), height=int(h), file_name=cp.name,
            license=0, flickr_url='', coco_url='', date_captured=0))
        annotations.append(dict(
            id=i, image_id=i, category_id=1, segmentation=[],
            area=float(w * h),
            # Full-image bbox: this is the project's convention (see
            # scripts/fix_bboxes_to_full_image.py) and what the model was
            # trained with. The per-image *_bbox.txt files are tight crops and
            # would introduce a train/inference mismatch if used here.
            bbox=[0, 0, int(w), int(h)],
            iscrowd=0,
            attributes=dict(occluded=False, keyframe=False),
            keypoints=flat, num_keypoints=n_labelled))

        perp, par = geometry_violations(np.asarray(kpts, dtype=np.float64),
                                        args.tolerance_deg)
        review.append(dict(
            file_name=cp.name,
            min_score=round(float(np.min(scores)), 4),
            mean_score=round(float(np.mean(scores)), 4),
            n_below_thr=int(np.sum(np.asarray(scores) < args.flag_score)),
            perp_violation_deg=round(perp, 2),
            parallel_violation_deg=round(par, 2),
            geom_violation_deg=round(max(perp, par), 2)))

        if n % 50 == 0 or n == len(color_paths):
            print(f'  {n}/{len(color_paths)}')

    coco = dict(
        licenses=[dict(name='', id=0, url='')],
        info=dict(contributor='', date_created='', description=
                  'Model pre-labels for human correction', url='',
                  version='', year=''),
        categories=[dict(id=1, name='Box_Flaps', supercategory='',
                         keypoints=KEYPOINT_NAMES, skeleton=SKELETON)],
        images=images, annotations=annotations)

    if existing_doc and args.merge_existing:
        # Human annotations are copied through untouched — including their
        # visibility flags and their tight bboxes. NOTE this leaves the file
        # with mixed bbox conventions (human=tight, model=full-image); run
        # scripts/fix_bboxes_to_full_image.py on it before training.
        coco['images'] = existing_doc['images'] + coco['images']
        coco['annotations'] = existing_doc['annotations'] + coco['annotations']
        print(f'Merged in {len(existing_doc["images"])} human-annotated images')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(coco, f, indent=1)

    # Triage list: geometry violations first (they catch the confident-but-wrong
    # corner swaps), then low confidence.
    review.sort(key=lambda r: (-r['geom_violation_deg'], r['min_score']))
    csv_path = out.with_name(out.stem + '_review.csv')
    with open(csv_path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(review[0].keys()))
        wr.writeheader()
        wr.writerows(review)

    n_geom = sum(1 for r in review if r['geom_violation_deg'] > 0)
    n_score = sum(1 for r in review if r['n_below_thr'] > 0)
    print(f'\nWrote {len(images)} images / {len(annotations)} annotations')
    print(f'  COCO json : {out}')
    print(f'  review csv: {csv_path}')
    print(f'\nTriage:')
    print(f'  {n_geom}/{len(review)} images violate the box geometry priors '
          f'(>{args.tolerance_deg:g} deg) -- review these FIRST')
    print(f'  {n_score}/{len(review)} images have >=1 keypoint below '
          f'score {args.flag_score:g}')
    print(f'\n  worst 5 by geometry:')
    for r in review[:5]:
        print(f'    {r["geom_violation_deg"]:6.1f} deg  min_score '
              f'{r["min_score"]:.2f}  {r["file_name"]}')


if __name__ == '__main__':
    main()
