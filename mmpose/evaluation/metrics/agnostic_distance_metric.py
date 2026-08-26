"""Score an unordered corner set against ground truth, per domain.

Companion to AgnosticUDPHeatmap. The model emits AT MOST 8 corners with no
identity, so index-wise error is meaningless -- predictions are matched to
ground truth by optimal assignment (Hungarian) and only then measured.

WHY THE MATCHING BUYS MORE THAN A FAIR SCORE
    The assignment INDUCES a labelling: once prediction j is matched to ground
    truth slot i, that prediction is, for measurement purposes, keypoint i. So
    every statistic the cardinal metric reports becomes computable on the
    agnostic model -- including corner-pair span error, which needs to know
    which two keypoints mark one physical corner. Without that, the two
    formulations could not be compared on the deliverable at all.

    The matching is optimal, so ANY permutation is free. That is the point: the
    gap between this metric on the agnostic model and the cardinal metric on
    the cardinal model is exactly what naming the corners costs.

Emits per domain:  <name>/mean_px, /median_px, /p90_px, /max_px,
                   /n_over_20px, /n_matched, /n_missing, /n_spurious,
                   /corner_span_err_px, /corner_span_bias_px
plus:              max/mean_px, max/corner_span_err_px

`n_missing` counts labelled corners with no prediction left to match --
the price of "at most 8" -- and `n_spurious` scored predictions that matched
nothing. Both are zero for a well-behaved cardinal model by construction,
which is why they are reported separately rather than folded into the error.
"""
import os.path as osp
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from scipy.optimize import linear_sum_assignment

from mmpose.registry import METRICS


@METRICS.register_module()
class AgnosticKeypointDistanceMetric(BaseMetric):
    """Hungarian-matched keypoint error, scored per deployment domain."""

    default_prefix: Optional[str] = None

    #: Same physical pairing the cardinal metric uses. Applied AFTER matching,
    #: on the induced labelling.
    CORNER_PAIRS = ((0, 7), (1, 2), (3, 4), (5, 6))

    def __init__(self,
                 domains: Sequence[dict],
                 score_thr: float = 0.0,
                 fail_thr_px: float = 20.0,
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
        self.domain_select: List[bool] = [
            bool(d.get('select', True)) for d in domains]
        self.score_thr = float(score_thr)
        self.fail_thr_px = float(fail_thr_px)
        self._roots = sorted(
            ((osp.normpath(d['data_root']), i) for i, d in enumerate(domains)),
            key=lambda t: -len(t[0]))

    def _route(self, img_path: str) -> int:
        p = osp.normpath(img_path)
        for root, idx in self._roots:
            if p.startswith(root + osp.sep) or p == root:
                return idx
        raise ValueError(f'sample {img_path!r} is under no configured data_root')

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        for s in data_samples:
            idx = self._route(s['img_path'])
            pred = np.asarray(s['pred_instances']['keypoints'][0], float)
            pscore = np.asarray(
                s['pred_instances'].get('keypoint_scores', [None])[0], float)
            gt = np.asarray(s['gt_instances']['keypoints'][0], float)
            gvis = np.asarray(s['gt_instances']['keypoints_visible'][0], float)

            keep_p = pscore > self.score_thr
            keep_g = gvis > 0
            pk, gk = pred[keep_p], gt[keep_g]
            gidx = np.nonzero(keep_g)[0]

            errs = np.full(len(gt), np.nan)
            if len(pk) and len(gk):
                cost = np.linalg.norm(pk[:, None, :] - gk[None, :, :], axis=-1)
                r, c = linear_sum_assignment(cost)
                for pi, gi in zip(r, c):
                    errs[gidx[gi]] = cost[pi, gi]

            # Induced labelling: matched_kpts[i] is the prediction that plays
            # the role of ground-truth keypoint i, or NaN where unmatched.
            matched = np.full((len(gt), 2), np.nan)
            if len(pk) and len(gk):
                for pi, gi in zip(r, c):
                    matched[gidx[gi]] = pk[pi]

            n_missing = int(np.isnan(errs[keep_g]).sum())
            n_spurious = int(max(0, len(pk) - min(len(pk), len(gk))))

            spans = []
            for a, b in self.CORNER_PAIRS:
                if (np.isnan(matched[a]).any() or np.isnan(matched[b]).any()
                        or not (keep_g[a] and keep_g[b])):
                    continue
                spans.append(float(
                    np.linalg.norm(matched[b] - matched[a]) -
                    np.linalg.norm(gt[b] - gt[a])))

            self.results.append(
                (idx, errs[~np.isnan(errs)], n_missing, n_spurious, spans))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger = MMLogger.get_current_instance()
        grouped: Dict[int, list] = {}
        for r in results:
            grouped.setdefault(r[0], []).append(r)

        metrics: Dict[str, float] = {}
        per_domain_mean: Dict[str, float] = {}
        per_domain_span: Dict[str, float] = {}
        for i, name in enumerate(self.domain_names):
            rows = grouped.get(i, [])
            if not rows:
                logger.warning(f'[AgnosticKeypointDistanceMetric] domain '
                               f'{name!r} received no samples')
                continue
            errs = np.concatenate([r[1] for r in rows]) if rows else np.array([])
            spans = np.array([s for r in rows for s in r[4]])
            if errs.size:
                metrics[f'{name}/mean_px'] = float(errs.mean())
                metrics[f'{name}/median_px'] = float(np.median(errs))
                metrics[f'{name}/p90_px'] = float(np.percentile(errs, 90))
                metrics[f'{name}/max_px'] = float(errs.max())
                metrics[f'{name}/n_over_20px'] = float(
                    (errs > self.fail_thr_px).sum())
                metrics[f'{name}/n_matched'] = float(errs.size)
                per_domain_mean[name] = metrics[f'{name}/mean_px']
            metrics[f'{name}/n_missing'] = float(sum(r[2] for r in rows))
            metrics[f'{name}/n_spurious'] = float(sum(r[3] for r in rows))
            if spans.size:
                metrics[f'{name}/corner_span_err_px'] = float(
                    np.median(np.abs(spans)))
                metrics[f'{name}/corner_span_bias_px'] = float(np.median(spans))
                per_domain_span[name] = metrics[f'{name}/corner_span_err_px']

        selectable = {n for n, s in zip(self.domain_names, self.domain_select)
                      if s}
        for key, src in (('mean_px', per_domain_mean),
                         ('corner_span_err_px', per_domain_span)):
            vals = {n: v for n, v in src.items() if n in selectable}
            if vals:
                w = max(vals, key=vals.get)
                metrics[f'max/{key}'] = vals[w]
                metrics[f'max/{key}_domain_index'] = float(
                    self.domain_names.index(w))

        if per_domain_mean:
            logger.info(
                '[AgnosticKeypointDistanceMetric] ' + '  '.join(
                    f'{n}={per_domain_mean[n]:.2f}px'
                    f'(span {metrics.get(f"{n}/corner_span_err_px", float("nan")):.2f},'
                    f' miss {metrics.get(f"{n}/n_missing", 0):.0f})'
                    for n in per_domain_mean) +
                f'  -> max={metrics.get("max/mean_px", float("nan")):.2f}px')
        return metrics
