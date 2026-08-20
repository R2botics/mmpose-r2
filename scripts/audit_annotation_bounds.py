"""Audit (and optionally fix) ground-truth keypoints that fall outside the image.

Why this matters here: mmpose maps BOTH COCO v=1 ("labelled but not visible")
and v=2 to visible (base_coco_style_dataset.py:296, `np.minimum(1, v)`). But
`UDPHeatmap.encode` sets keypoint_weight=0 for a target outside the heatmap, so
such a point is NEVER TRAINED while still being SCORED -- the model is judged on
corners it was explicitly taught to ignore. In this project that alone accounted
for 26 of RSC val's 43 "unusable corners".

The application accepts partial views: when a flap corner is genuinely outside
the frame, predicting nothing is the correct behaviour. `v=0` is COCO's way of
saying exactly that, so `--fix` rewrites out-of-bounds v=1/v=2 to v=0 and
recomputes num_keypoints. That makes the annotations agree with what training
already does, and with what the evaluation now does.

Groups by capture session (parsed from the filename timestamp) so a convention
that changed partway through a collection shows up as a block rather than noise.

Usage:
    # report only
    python scripts/audit_annotation_bounds.py data/RSC_Keypoints_RGBD/annotations/*.json

    # allow a few px of overshoot (those still train -- the Gaussian is partly
    # visible), then rewrite the rest to v=0, keeping a .bak
    python scripts/audit_annotation_bounds.py --margin 5 --fix path/to/ann.json
"""
import argparse
import collections
import json
import re
import shutil
from pathlib import Path

TS = re.compile(r'_(\d{4}-\d{2}-\d{2})T(\d{2})-')


def session_of(file_name: str) -> str:
    """Capture session = date + hour from the filename, else 'unknown'."""
    m = TS.search(file_name)
    return f'{m.group(1)} {m.group(2)}h' if m else 'unknown'


def audit(path: Path, margin: float):
    data = json.loads(path.read_text())
    imgs = {i['id']: i for i in data['images']}
    names = next((c['keypoints'] for c in data.get('categories', [])
                  if c.get('keypoints')), None)

    rows = []
    for ann in data['annotations']:
        im = imgs.get(ann['image_id'])
        if im is None:
            continue
        w, h = im['width'], im['height']
        kp = ann['keypoints']
        for i in range(0, len(kp), 3):
            x, y, v = kp[i], kp[i + 1], kp[i + 2]
            over = max(-x, x - (w - 1), -y, y - (h - 1), 0.0)
            rows.append(dict(ann=ann, slot=i, v=int(v), over=over,
                             name=(names[i // 3] if names else f'kpt_{i//3}'),
                             file=im['file_name'], w=w, h=h, x=x, y=y))
    return data, rows


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument('files', nargs='+')
    ap.add_argument('--margin', type=float, default=0.0,
                    help='px of overshoot tolerated before a point counts as '
                         'out of bounds (default 0). A small value keeps '
                         'near-edge corners that DO still train.')
    ap.add_argument('--fix', action='store_true',
                    help='rewrite out-of-bounds v=1/v=2 to v=0 and recompute '
                         'num_keypoints')
    ap.add_argument('--no-backup', action='store_true')
    ap.add_argument('--fix-convention', action='store_true',
                    help='Roll the keypoints of convention-violating (90deg '
                         'clocked) annotations by two slots so the SHORT flaps '
                         'become NORTH/SOUTH. Implies --check-convention.')
    ap.add_argument('--check-convention', action='store_true',
                    help='Also report annotations that violate the flap '
                         'convention (NORTH/SOUTH = the two SHORT minor flaps, '
                         'EAST/WEST = the two LONG major flaps). A violation is '
                         'a 90-degree clocked label, which -- unlike an '
                         'out-of-bounds point -- carries FULL gradient during '
                         'training and actively teaches the wrong assignment.')
    args = ap.parse_args()

    for f in args.files:
        path = Path(f)
        data, rows = audit(path, args.margin)
        oob = [r for r in rows if r['v'] > 0 and r['over'] > args.margin]
        labelled = [r for r in rows if r['v'] > 0]

        print(f'\n=== {path} ===')
        print(f'  keypoints: {len(rows)} total, {len(labelled)} labelled (v>0)')
        byv = collections.Counter(r['v'] for r in rows)
        print(f'  visibility flags: {dict(sorted(byv.items()))}')
        if not oob:
            print('  no labelled keypoints outside the image  -- nothing to do')
            continue

        oob_by_v = collections.Counter(r['v'] for r in oob)
        print(f'  OUT OF BOUNDS (beyond {args.margin:g}px): {len(oob)} '
              f'({100*len(oob)/max(len(labelled),1):.1f}% of labelled)')
        print(f'    by visibility flag: {dict(sorted(oob_by_v.items()))}'
              '   <- v=1 here is the "annotated outside the frame" style')
        ov = sorted(r['over'] for r in oob)
        print(f'    overshoot px: median {ov[len(ov)//2]:.1f}  max {ov[-1]:.1f}')

        # does the convention track the capture session?
        per = collections.defaultdict(lambda: [0, 0])
        for r in labelled:
            s = session_of(r['file'])
            per[s][1] += 1
            if r['over'] > args.margin:
                per[s][0] += 1
        print(f'\n  {"capture session":<20}{"labelled":>10}{"out of bounds":>15}{"rate":>8}')
        for s in sorted(per):
            n_oob, n_tot = per[s]
            flag = '  <<<' if n_tot and n_oob / n_tot > 0.10 else ''
            print(f'  {s:<20}{n_tot:>10}{n_oob:>15}{100*n_oob/max(n_tot,1):>7.1f}%{flag}')

        byname = collections.Counter(r['name'] for r in oob)
        print(f'\n  by keypoint: {dict(byname.most_common())}')

        if args.check_convention or args.fix_convention:
            imgs = {i['id']: i for i in data['images']}
            PAIRS = {'NORTH': (0, 1), 'EAST': (2, 3), 'SOUTH': (4, 5), 'WEST': (6, 7)}
            viol, near, ok = [], 0, 0
            for ann in data['annotations']:
                im = imgs.get(ann['image_id'])
                if im is None:
                    continue
                k = [ann['keypoints'][i:i + 3] for i in range(0, len(ann['keypoints']), 3)]
                span = {}
                for nm, (a_, b_) in PAIRS.items():
                    if k[a_][2] > 0 and k[b_][2] > 0:
                        span[nm] = ((k[a_][0] - k[b_][0]) ** 2 +
                                    (k[a_][1] - k[b_][1]) ** 2) ** 0.5
                if len(span) < 4:
                    continue
                ns = (span['NORTH'] + span['SOUTH']) / 2
                ew = (span['EAST'] + span['WEST']) / 2
                if ns >= ew:
                    viol.append((ns / ew, im['file_name'], span, ann))
                elif abs(ns / ew - 1) < 0.10:
                    near += 1
                else:
                    ok += 1
            total = len(viol) + near + ok
            print(f'\n  convention check (N/S should be the SHORT flaps):')
            print(f'    fully-labelled boxes: {total}')
            print(f'    respects convention : {ok}')
            print(f'    near-square (ambiguous, within 10%): {near}')
            print(f'    VIOLATIONS (90deg clocked): {len(viol)}'
                  f'  = {100*len(viol)/max(total,1):.1f}%')
            for r, fn, sp, _a in sorted(viol, key=lambda t: -t[0])[:10]:
                print(f'      ratio {r:.2f}  N={sp["NORTH"]:.0f} S={sp["SOUTH"]:.0f} | '
                      f'E={sp["EAST"]:.0f} W={sp["WEST"]:.0f}   {fn[:52]}')
            if viol:
                print('    -> these carry FULL gradient in training, unlike '
                      'out-of-bounds points (weight 0).')

            if args.fix_convention and viol:
                if not args.no_backup:
                    bak = path.with_suffix(path.suffix + '.bak')
                    if not bak.exists():
                        shutil.copy(path, bak)
                        print(f'    backup written: {bak}')
                    else:
                        print(f'    backup kept (already exists): {bak}')
                for _r, fn, _sp, ann in viol:
                    kp = ann['keypoints']
                    trip = [kp[i:i + 3] for i in range(0, len(kp), 3)]
                    # roll right by 2 slots: new NORTH = old WEST, so the SHORT
                    # pair lands on NORTH/SOUTH as the convention requires
                    rolled = trip[-2:] + trip[:-2]
                    ann['keypoints'] = [v for t in rolled for v in t]
                path.write_text(json.dumps(data))
                print(f'    FIXED CONVENTION: rolled {len(viol)} annotations '
                      f'by two flaps -> {path}')

        if args.fix:
            if not args.no_backup:
                bak = path.with_suffix(path.suffix + '.bak')
                if not bak.exists():
                    shutil.copy(path, bak)
                    print(f'\n  backup written: {bak}')
                else:
                    print(f'\n  backup kept (already exists, NOT overwritten): {bak}')
            for r in oob:
                r['ann']['keypoints'][r['slot'] + 2] = 0
            for ann in data['annotations']:
                kp = ann['keypoints']
                ann['num_keypoints'] = sum(1 for i in range(2, len(kp), 3)
                                           if kp[i] > 0)
            path.write_text(json.dumps(data))
            print(f'  FIXED: {len(oob)} keypoints set to v=0, '
                  f'num_keypoints recomputed -> {path}')
        else:
            print('\n  (report only -- pass --fix to rewrite these to v=0)')


if __name__ == '__main__':
    main()
