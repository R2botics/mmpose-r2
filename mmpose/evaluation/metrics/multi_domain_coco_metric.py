"""COCO keypoint metric that scores several deployment domains separately.

Motivation: when a model must work on more than one camera rig, a single
merged validation number hides the thing you most need to see. A checkpoint
that gains on one rig and loses on another averages out to "fine", so
selecting on the mean silently accepts regressions. Scoring each domain and
selecting on the WORST one ("works everywhere") cannot be gamed that way — it
only improves when the weakest domain improves.

Emits, per evaluation:

    <domain>/coco/AP ...    one full CocoMetric result set per domain
    min/coco/AP             the lowest per-domain AP  <- use for save_best
    min/domain_index        which domain was binding (index into `domains`)

Usage:

    val_evaluator = dict(
        type='MultiDomainCocoMetric',
        domains=[
            dict(name='rsc',   data_root='data/RSC_Keypoints_RGBD',
                 ann_file='data/RSC_Keypoints_RGBD/annotations/person_keypoints_val.json'),
            dict(name='chewy', data_root='data/ChewyCrops',
                 ann_file='data/ChewyCrops/annotations/person_keypoints_Train.json'),
        ])

    default_hooks = dict(checkpoint=dict(
        save_best=['min/coco/AP'], rule='greater'))

    # ...with a CombinedDataset val_dataloader unioning the same two sets.

Samples are routed to a domain by matching `img_path` against `data_root`,
NOT by `img_id`: separate COCO files routinely reuse ids 1..N, so routing by
id would mix domains. Matching uses the LONGEST matching root, so nested names
like data/ChewyCrops and data/ChewyCrops2 are disambiguated correctly.
"""
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from .coco_metric import CocoMetric


@METRICS.register_module()
class MultiDomainCocoMetric(BaseMetric):
    """Run one CocoMetric per domain and report each plus their minimum.

    Args:
        domains (list[dict]): One entry per deployment domain, each with
            ``name``, ``data_root`` and ``ann_file``.
        collect_device (str): Device for distributed result collection.
        prefix (str, optional): Left as None — this metric emits fully
            qualified keys (``rsc/coco/AP``) so they are readable as-is.
        **coco_kwargs: Forwarded verbatim to every underlying CocoMetric
            (e.g. ``use_area``, ``score_mode``, ``nms_mode``), so all domains
            are scored identically.
    """

    default_prefix: Optional[str] = None

    def __init__(self,
                 domains: Sequence[dict],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **coco_kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if not domains:
            raise ValueError('`domains` must list at least one domain')
        for d in domains:
            missing = {'name', 'data_root', 'ann_file'} - set(d)
            if missing:
                raise ValueError(f'domain entry {d} is missing {sorted(missing)}')

        self.domain_names: List[str] = [d['name'] for d in domains]
        if len(set(self.domain_names)) != len(self.domain_names):
            raise ValueError(f'duplicate domain names: {self.domain_names}')
        # `select=False` reports a domain but keeps it out of min/coco/AP, so
        # it cannot drive save_best -- see MultiDomainKeypointDistanceMetric.
        self.domain_select: List[bool] = [
            bool(d.get('select', True)) for d in domains]
        if not any(self.domain_select):
            raise ValueError('at least one domain must have select=True, '
                             'otherwise min/coco/AP is never emitted')

        # Longest root first so nested roots (data/ChewyCrops vs
        # data/ChewyCrops2) route to the more specific one.
        self._roots = sorted(
            ((osp.normpath(d['data_root']), i) for i, d in enumerate(domains)),
            key=lambda t: -len(t[0]))

        self.sub_metrics: List[CocoMetric] = [
            CocoMetric(ann_file=d['ann_file'], **coco_kwargs) for d in domains
        ]

    # ------------------------------------------------------------------ meta
    @property
    def dataset_meta(self) -> Optional[dict]:
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Forward to every sub-metric — CocoMetric needs it for OKS sigmas."""
        self._dataset_meta = dataset_meta
        for m in self.sub_metrics:
            m.dataset_meta = dataset_meta

    # ----------------------------------------------------------------- route
    def _route(self, img_path: str) -> int:
        p = osp.normpath(img_path)
        for root, idx in self._roots:
            if p.startswith(root + osp.sep) or p == root:
                return idx
        raise ValueError(
            f'sample {img_path!r} does not live under any configured '
            f'data_root ({[r for r, _ in self._roots]}). Every val sample must '
            'belong to exactly one domain.')

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Delegate to the owning domain's CocoMetric, tagging each result.

        Reuses CocoMetric.process verbatim rather than duplicating its parsing,
        then moves what it appended into our own results list so distributed
        collection still sees a single flat list.
        """
        for data_sample in data_samples:
            idx = self._route(data_sample['img_path'])
            sub = self.sub_metrics[idx]
            n_before = len(sub.results)
            sub.process(data_batch, [data_sample])
            for item in sub.results[n_before:]:
                self.results.append((idx, item))
            del sub.results[n_before:]

    # --------------------------------------------------------------- compute
    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        grouped = defaultdict(list)
        for idx, item in results:
            grouped[idx].append(item)

        metrics: Dict[str, float] = {}
        per_domain_ap: Dict[str, float] = {}

        for idx, name in enumerate(self.domain_names):
            items = grouped.get(idx, [])
            if not items:
                logger.warning(
                    f'[MultiDomainCocoMetric] domain {name!r} received no '
                    'samples — check that the val dataloader includes it.')
                continue
            sub = self.sub_metrics[idx]
            logger.info(f'[MultiDomainCocoMetric] scoring {name} '
                        f'({len(items)} samples)')
            res = sub.compute_metrics(items)
            for k, v in res.items():
                metrics[f'{name}/coco/{k}'] = v
            if 'AP' in res:
                per_domain_ap[name] = res['AP']

        selectable = {n for n, s in zip(self.domain_names, self.domain_select)
                      if s}
        sel_ap = {n: v for n, v in per_domain_ap.items() if n in selectable}
        if sel_ap:
            worst = min(sel_ap, key=sel_ap.get)
            metrics['min/coco/AP'] = sel_ap[worst]
            metrics['min/domain_index'] = float(self.domain_names.index(worst))
            summary = '  '.join(
                f'{k}={v:.4f}' + ('' if k in selectable else '*')
                for k, v in per_domain_ap.items())
            logger.info(f'[MultiDomainCocoMetric] {summary}  '
                        f'-> min={per_domain_ap[worst]:.4f} (binding: {worst})')
        return metrics
