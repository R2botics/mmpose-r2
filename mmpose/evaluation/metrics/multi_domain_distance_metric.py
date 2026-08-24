"""Raw pixel-distance keypoint metric, scored per deployment domain.

Why pixels rather than OKS/AP or a size-normalised error:

  * The model emits pixel coordinates and downstream converts to physical
    units itself, using the depth map. A metric that divides by image size
    applies a scale correction downstream already applies properly, per image
    — double-correcting. Evaluate in the space the model is responsible for.
  * coco/AP is blind at this accuracy level. Because this project rewrites
    bboxes to full-image, OKS takes s = sqrt(image area) (296-725 px on the
    Chewy val set), so even the strictest threshold in AP@0.50:0.95 tolerates
    ~33 px of error while the model's median error is ~3 px. AP saturates
    near 1.0 and cannot separate checkpoints.
  * Dividing by image diagonal over-corrects: crop size spans ~2.5x on this
    data, but flap height (216-586 mm) implies far less variation in actual
    camera-to-corner distance.

Selection uses `max/mean_px` with rule='less' — the checkpoint whose WORST
domain has the lowest pixel error. Note the criterion inverts relative to an
AP-style metric: "works everywhere" means minimise the maximum error.

`mean_px` drives selection deliberately: for corner-to-corner metrology a
large error does not degrade a measurement, it destroys it, and a median
would shrug those off. `median_px` is logged alongside so you can tell why
`mean_px` moved — mean up with median flat means new catastrophic failures;
both up means broad degradation.

Unlabelled ground-truth keypoints (visibility 0) are excluded, not scored as
zero error. That matters here: on this data the SOUTH flap is often not
visible, so a large fraction of val keypoints are v=0.

Out-of-image ground truth is excluded too (`ignore_out_of_bounds`). The RSC
annotations mark corners that fall outside the crop as **v=1**, and mmpose maps
both v=1 and v=2 to "visible" (base_coco_style_dataset.py:296,
`np.minimum(1, v)`), so they would otherwise be scored. They cannot be: a
prediction is confined to the heatmap's extent, while these targets sit up to
~210 px outside the image. Worse, `UDPHeatmap.encode` sets their
`keypoint_weight` to 0, so the model is never TRAINED on them -- scoring them
judges the model on corners it was explicitly taught to ignore. Left in, they
pin `max_px` to a constant and freeze the failure count regardless of how well
training goes.

Emits per domain:  <name>/mean_px, /median_px, /p90_px, /max_px,
                   /n_over_20px, /frac_over_20px, /n_keypoints
plus:              max/mean_px, max/domain_index

Set ``report_worst`` to log the identity (image + keypoint name) of the
largest errors each evaluation. A max error that is IDENTICAL across many
epochs is the signature of a mislabelled ground-truth point rather than a
model failure -- the prediction tracks the real corner while the annotation
stays put, so the gap does not move as the weights do. Naming the file and
corner turns that from a number into something you can go and look at.

Usage:
    val_evaluator = [
        dict(type='MultiDomainKeypointDistanceMetric',
             domains=[dict(name='rsc',   data_root='data/RSC_Keypoints_RGBD'),
                      dict(name='chewy', data_root='data/ChewyCrops')],
             fail_thr_px=20.0),
        dict(type='MultiDomainCocoMetric', domains=[...]),   # optional, for continuity
    ]
    default_hooks = dict(checkpoint=dict(
        save_best=['max/mean_px'], rule='less'))
"""
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS


@METRICS.register_module()
class MultiDomainKeypointDistanceMetric(BaseMetric):
    """Per-domain Euclidean keypoint error in pixels.

    Args:
        domains (list[dict]): One entry per deployment domain, each with
            ``name`` and ``data_root``. Samples are routed by ``img_path``
            against ``data_root`` (NOT by ``img_id``: separate COCO files
            routinely reuse ids 1..N). Longest match wins, so nested roots
            like data/ChewyCrops and data/ChewyCrops2 disambiguate correctly.
        fail_thr_px (float): Errors at or above this count as unusable
            corners. Defaults to 20.0.
        ignore_out_of_bounds (bool): Skip ground-truth keypoints lying outside
            the image, matching what training already does. Defaults to True.
        oob_margin_px (float): Slack before a GT keypoint counts as out of
            bounds. Defaults to 0.0. Mild overshoots still train (the Gaussian
            is partly visible), so a small positive margin keeps those scored.
        report_worst (int): Log this many worst-offending keypoints per domain
            after each evaluation, with image path and keypoint name. 0
            disables. Defaults to 5.
        report_cyclic (bool): Also score each instance under the four cyclic
            re-assignments of the flaps (NORTH->EAST->SOUTH->WEST->NORTH) and
            report the best. On a rotated box the cardinal labels can be one
            flap out, which makes every corner land on its neighbour's position
            -- roughly one box side away. That inflates the error by ~100px per
            corner while the predicted GEOMETRY is actually correct. Comparing
            `mean_px` with `cyclic_mean_px` separates "wrong shape" from
            "right shape, rotated labels".

            This project's downstream associates the four spans across two
            camera views, so a rotated/clocked assignment is harmless there --
            which makes `cyclic_mean_px` the error that actually matters and
            `max/cyclic_mean_px` the right thing to select on. Both are always
            emitted; pick via `save_best`. Defaults to True.
    """

    default_prefix: Optional[str] = None

    def __init__(self,
                 domains: Sequence[dict],
                 fail_thr_px: float = 20.0,
                 ignore_out_of_bounds: bool = True,
                 oob_margin_px: float = 0.0,
                 report_worst: int = 5,
                 report_cyclic: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if not domains:
            raise ValueError('`domains` must list at least one domain')
        for d in domains:
            missing = {'name', 'data_root'} - set(d)
            if missing:
                raise ValueError(f'domain entry {d} is missing {sorted(missing)}')

        self.domain_names: List[str] = [d['name'] for d in domains]
        if len(set(self.domain_names)) != len(self.domain_names):
            raise ValueError(f'duplicate domain names: {self.domain_names}')
        # `select=False` reports a domain but keeps it out of the max/
        # aggregation, so it cannot drive save_best. That is what a held-out
        # SYNTHETIC split needs: it tells you whether the run is overfitting
        # synthetic, and it must never decide which checkpoint ships.
        self.domain_select: List[bool] = [
            bool(d.get('select', True)) for d in domains]
        if not any(self.domain_select):
            raise ValueError('at least one domain must have select=True, '
                             'otherwise max/* is never emitted')
        self.fail_thr_px = float(fail_thr_px)
        self.ignore_out_of_bounds = bool(ignore_out_of_bounds)
        self.oob_margin_px = float(oob_margin_px)
        self.report_worst = int(report_worst)
        self.report_cyclic = bool(report_cyclic)
        self._roots = sorted(
            ((osp.normpath(d['data_root']), i) for i, d in enumerate(domains)),
            key=lambda t: -len(t[0]))

    #: The four PHYSICAL box corners. Each is marked by two keypoints -- one
    #: from an N/S flap and one from an E/W flap -- and the model routinely
    #: collapses the two onto a single point. Measured on Chewy+OOD: 6% of
    #: pairs collapse (predicted separation < half of GT) yet carry 16% of all
    #: error, because collapse rate rises with true separation (4% under 10px,
    #: 22% over 40px). The cause is resolution: median separation is 1.89
    #: heatmap px against sigma=2, so 54% of training pairs are closer than one
    #: sigma and the two target blobs are nearly the same blob.
    #:
    #: Nothing else in this metric sees it -- a collapsed pair just looks like
    #: two mediocre keypoints -- which is why it needs its own counter.
    CORNER_PAIRS = ((0, 7), (1, 2), (3, 4), (5, 6))

    def _route(self, img_path: str) -> int:
        p = osp.normpath(img_path)
        for root, idx in self._roots:
            if p.startswith(root + osp.sep) or p == root:
                return idx
        raise ValueError(
            f'sample {img_path!r} does not live under any configured '
            f'data_root ({[r for r, _ in self._roots]}).')

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        n_dropped = 0
        for data_sample in data_samples:
            idx = self._route(data_sample['img_path'])
            pred = data_sample['pred_instances']['keypoints']       # (N, K, 2)
            gt = data_sample['gt_instances']['keypoints']           # (N, K, 2)
            vis = data_sample['gt_instances'].get('keypoints_visible')
            if vis is None:
                vis = np.ones(gt.shape[:2], dtype=bool)
            vis = np.asarray(vis)
            if vis.ndim == 3:            # some transforms carry a trailing dim
                vis = vis[..., 0]

            n = min(len(pred), len(gt))
            # ori_shape is (H, W) of the original image the GT lives in
            shape = data_sample.get('ori_shape')
            for i in range(n):
                m = np.asarray(vis[i]) > 0
                if self.ignore_out_of_bounds and shape is not None:
                    h_img, w_img = shape[0], shape[1]
                    g = np.asarray(gt[i])[:, :2]
                    e = self.oob_margin_px
                    inside = ((g[:, 0] >= -e) & (g[:, 0] <= w_img - 1 + e) &
                              (g[:, 1] >= -e) & (g[:, 1] <= h_img - 1 + e))
                    n_dropped += int((m & ~inside).sum())
                    m = m & inside
                if not m.any():
                    continue
                pk = np.asarray(pred[i])[:, :2]
                gk = np.asarray(gt[i])[:, :2]
                d = np.linalg.norm(pk[m] - gk[m], axis=-1)
                # keep which keypoint each distance came from, so the worst
                # offenders can be named rather than just counted
                kpt_idx = np.flatnonzero(m)

                # best of the four cyclic flap assignments. Each flap owns two
                # consecutive keypoints, so one flap of rotation is a shift of
                # two indices.
                best_d, best_shift = d, 0
                if self.report_cyclic and pk.shape[0] == 8:
                    for shift in (2, 4, 6):
                        rolled = np.roll(pk, shift, axis=0)
                        cand = np.linalg.norm(rolled[m] - gk[m], axis=-1)
                        if cand.mean() < best_d.mean():
                            best_d, best_shift = cand, shift

                # corner-pair separation: (gt_sep, pred_sep, summed pair error)
                pairs = []
                if pk.shape[0] == 8:
                    for a, b in self.CORNER_PAIRS:
                        if not (m[a] and m[b]):
                            continue
                        gsep = float(np.linalg.norm(gk[b] - gk[a]))
                        psep = float(np.linalg.norm(pk[b] - pk[a]))
                        perr = float(np.linalg.norm(pk[a] - gk[a]) +
                                     np.linalg.norm(pk[b] - gk[b]))
                        pairs.append((gsep, psep, perr))

                self.results.append((idx, d.astype(np.float64), kpt_idx,
                                     data_sample['img_path'],
                                     best_d.astype(np.float64), best_shift,
                                     n_dropped, pairs))
                n_dropped = 0

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        grouped = defaultdict(list)
        offenders = defaultdict(list)   # (error, img_path, keypoint index)
        cyc = defaultdict(list)
        shifts = defaultdict(list)
        dropped = defaultdict(int)
        pairs_by_domain = defaultdict(list)
        for idx, d, kpt_idx, img_path, best_d, best_shift, n_oob, prs in results:
            dropped[idx] += n_oob
            pairs_by_domain[idx].extend(prs)
            grouped[idx].append(d)
            cyc[idx].append(best_d)
            shifts[idx].append(best_shift)
            if self.report_worst:
                for e, ki in zip(d, kpt_idx):
                    offenders[idx].append((float(e), img_path, int(ki)))

        kpt_names = None
        if self.dataset_meta:
            kpt_names = self.dataset_meta.get('keypoint_id2name')

        metrics: Dict[str, float] = {}
        per_domain_mean: Dict[str, float] = {}

        for idx, name in enumerate(self.domain_names):
            chunks = grouped.get(idx, [])
            if not chunks:
                logger.warning(
                    f'[MultiDomainKeypointDistanceMetric] domain {name!r} '
                    'received no samples — check the val dataloader.')
                continue
            e = np.concatenate(chunks)
            over = float((e >= self.fail_thr_px).sum())
            metrics.update({
                f'{name}/mean_px': float(e.mean()),
                f'{name}/median_px': float(np.median(e)),
                f'{name}/p90_px': float(np.percentile(e, 90)),
                f'{name}/max_px': float(e.max()),
                f'{name}/n_over_{int(self.fail_thr_px)}px': over,
                f'{name}/frac_over_{int(self.fail_thr_px)}px': float(over / e.size),
                f'{name}/n_keypoints': float(e.size),
            })
            per_domain_mean[name] = float(e.mean())

            # Collapsed corner pairs. `pair_sep_ratio` is the headline: 1.0
            # means the model reproduces the true corner separation, and it
            # sits near 0.72 today because predictions shrink toward the
            # typical ~10px separation regardless of the actual box.
            prs = pairs_by_domain.get(idx, [])
            if prs:
                gsep = np.array([p[0] for p in prs])
                psep = np.array([p[1] for p in prs])
                perr = np.array([p[2] for p in prs])
                collapsed = psep < 0.5 * gsep
                ok = gsep > 1e-6
                metrics.update({
                    f'{name}/n_corner_pairs': float(len(prs)),
                    f'{name}/n_collapsed_pairs': float(collapsed.sum()),
                    f'{name}/frac_collapsed_pairs': float(collapsed.mean()),
                    # MEDIAN, not mean: pairs with a tiny true separation give
                    # huge pred/gt ratios and drag the mean above 1.0 even
                    # while the model is systematically shrinking separations.
                    f'{name}/pair_sep_ratio': float(
                        np.median(psep[ok] / gsep[ok])) if ok.any() else 0.0,
                    # Slope of predicted-vs-true separation. This is the one
                    # that shows the shrinkage: 1.0 = faithful, and it sits
                    # near 0.72 today because predictions regress toward the
                    # typical ~10px corner separation whatever the box does.
                    f'{name}/pair_sep_slope': float(
                        np.polyfit(gsep[ok], psep[ok], 1)[0])
                    if ok.sum() > 1 else 0.0,
                    f'{name}/collapsed_err_share': float(
                        perr[collapsed].sum() / perr.sum())
                    if perr.sum() > 0 else 0.0,
                })
            if self.ignore_out_of_bounds:
                metrics[f'{name}/n_gt_out_of_bounds'] = float(dropped[idx])
                if dropped[idx]:
                    logger.info(
                        f'[MultiDomainKeypointDistanceMetric] {name}: skipped '
                        f'{dropped[idx]} ground-truth keypoints lying outside '
                        'the image (the model is not trained on those either)')

            if self.report_cyclic and cyc[idx]:
                ce = np.concatenate(cyc[idx])
                sh = np.asarray(shifts[idx])
                n_rot = int((sh != 0).sum())
                metrics[f'{name}/cyclic_mean_px'] = float(ce.mean())
                metrics[f'{name}/cyclic_n_over_{int(self.fail_thr_px)}px'] = \
                    float((ce >= self.fail_thr_px).sum())
                metrics[f'{name}/n_rotated_instances'] = float(n_rot)
                # Split by KIND of rotation. This project assigns NORTH/SOUTH
                # to the two SHORT (minor) flaps and EAST/WEST to the two LONG
                # (major wall) flaps, which is rotation-invariant. So:
                #   shift 4 (180 deg) -- swaps N<->S and E<->W. Short stays
                #     short: the convention is respected and only the
                #     which-end-is-north tie-break differs. Benign.
                #   shift 2 or 6 (90 deg) -- swaps a short flap for a long
                #     one. On a clearly rectangular box that means the aspect
                #     orientation was misread; on a SQUARE box it is genuinely
                #     undecidable and the model's guess is as good as any.
                #     Either way this project's downstream associates the four
                #     spans across the two camera views, so it is absorbed --
                #     which is why selection uses max/cyclic_mean_px (best of
                #     all four assignments) rather than the raw error.
                #     Treat a rising n_clock90 on rectangular boxes as a signal
                #     worth looking at, not as a failure to fix.
                metrics[f'{name}/n_flip180'] = float((sh == 4).sum())
                metrics[f'{name}/n_clock90'] = float(((sh == 2) | (sh == 6)).sum())
                logger.info(
                    f'[MultiDomainKeypointDistanceMetric] {name} rotation kinds: '
                    f'{int((sh == 4).sum())} x 180deg flip (benign, short flaps '
                    f'stay short), '
                    f'{int(((sh == 2) | (sh == 6)).sum())} x 90deg clock '
                    f'(major/minor swapped -- violates the N/S=short convention)')
                logger.info(
                    f'[MultiDomainKeypointDistanceMetric] {name} cyclic check: '
                    f'{n_rot}/{len(sh)} instances score better under a rotated '
                    f'flap assignment; mean {e.mean():.2f}px -> '
                    f'{ce.mean():.2f}px, >{int(self.fail_thr_px)}px '
                    f'{int((e >= self.fail_thr_px).sum())} -> '
                    f'{int((ce >= self.fail_thr_px).sum())}')

            if self.report_worst and offenders[idx]:
                top = sorted(offenders[idx], key=lambda t: -t[0])[:self.report_worst]
                logger.info(f'[MultiDomainKeypointDistanceMetric] {name} '
                            f'worst {len(top)} keypoints:')
                for err, img_path, ki in top:
                    kn = (kpt_names.get(ki, f'kpt_{ki}')
                          if isinstance(kpt_names, dict) else f'kpt_{ki}')
                    logger.info(f'    {err:8.1f}px  {kn:<9s} '
                                f'{osp.basename(img_path)}')

        # Only selectable domains take part in the minimax.
        selectable = {n for n, s in zip(self.domain_names, self.domain_select)
                      if s}
        sel_mean = {n: v for n, v in per_domain_mean.items() if n in selectable}
        if sel_mean:
            worst = max(sel_mean, key=sel_mean.get)
            metrics['max/mean_px'] = sel_mean[worst]
            metrics['max/domain_index'] = float(self.domain_names.index(worst))

            # same minimax, but on the rotation-tolerant error
            cyc_means = {n: metrics[f'{n}/cyclic_mean_px']
                         for n in sel_mean
                         if f'{n}/cyclic_mean_px' in metrics}
            if cyc_means:
                cworst = max(cyc_means, key=cyc_means.get)
                metrics['max/cyclic_mean_px'] = cyc_means[cworst]
                metrics['max/cyclic_domain_index'] = float(
                    self.domain_names.index(cworst))
            summary = '  '.join(
                f'{n}={metrics[f"{n}/mean_px"]:.2f}px'
                f'(med {metrics[f"{n}/median_px"]:.2f},'
                f' >{int(self.fail_thr_px)}px {int(metrics[f"{n}/n_over_{int(self.fail_thr_px)}px"])})'
                for n in per_domain_mean)
            logger.info(
                f'[MultiDomainKeypointDistanceMetric] {summary}  '
                f'-> max={per_domain_mean[worst]:.2f}px (binding: {worst})')
        return metrics
