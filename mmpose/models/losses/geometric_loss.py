"""Geometric keypoint MSE loss for box flap detection.

Augments standard `KeypointMSELoss` with two soft geometric priors that hold
for a (near) top-down camera on a rectangular box, regardless of how the flaps
are bent up around their hinge axes:

  * PERPENDICULARITY between adjacent flap edges (NORTH ⊥ EAST etc.)
  * PARALLELISM between opposite flap edges (NORTH ∥ SOUTH etc.)
  * SPAN: predicted flap length matches the labelled flap length. Optional
    (`span_weight=0.0` by default). The two angular terms normalise their edge
    vectors and so cannot see length at all -- see the note in forward().

Both penalties have a `tolerance_deg` "free zone" so small perspective
distortion (e.g. ~10° camera tilt) doesn't get penalised. Keypoints are
extracted from the predicted heatmaps via differentiable soft-argmax so the
geometric loss truly trains the heatmap weights.
"""
from math import cos, radians
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmpose.registry import MODELS
from .heatmap_loss import KeypointMSELoss


@MODELS.register_module()
class GeometricKeypointMSELoss(KeypointMSELoss):
    """KeypointMSELoss + tolerance-aware perpendicularity and parallelism.

    Args:
        perp_weight:        Weight on the perpendicularity term.
        parallel_weight:    Weight on the parallelism term.
        span_weight:        Weight on the flap-length term. Defaults to 0.0
                            (disabled) so existing configs are unchanged.
                            The other two terms constrain edge direction only;
                            this one constrains edge LENGTH, which is what
                            downstream actually measures.
        span_tolerance:     Relative slack on flap length (0.01 = 1%) before
                            the span term incurs any penalty. Mirrors the
                            `tolerance_deg` free zone of the angular terms.
        tolerance_deg:      Angular slack (degrees) before each term incurs
                            any penalty. Keep at ~10 for slight perspective.
                            Measured max in the labels is 8.8 deg, so 10 has
                            little headroom -- widen if production sees more
                            steeply angled views than the training data.
        softmax_temp:       Temperature for soft-argmax. Larger = sharper
                            peak (closer to true argmax). 10.0 works well.
        perpendicular_pairs: Edge-pairs that should be perpendicular, each
                            ((idx_a, idx_b), (idx_c, idx_d)) means edge AB
                            should be perpendicular to edge CD. Defaults to
                            the four NORTH/SOUTH × EAST/WEST combinations.
        parallel_pairs:     Edge-pairs that should be parallel. Defaults to
                            NORTH∥SOUTH and EAST∥WEST.
        **kwargs:           Forwarded to `KeypointMSELoss`.
    """

    DEFAULT_PERPENDICULAR_PAIRS = (
        ((0, 1), (2, 3)),  # NORTH (N1-N2) ⊥ EAST  (E1-E2)
        ((0, 1), (6, 7)),  # NORTH          ⊥ WEST  (W1-W2)
        ((4, 5), (2, 3)),  # SOUTH (S1-S2) ⊥ EAST
        ((4, 5), (6, 7)),  # SOUTH          ⊥ WEST
    )
    DEFAULT_PARALLEL_PAIRS = (
        ((0, 1), (4, 5)),  # NORTH ∥ SOUTH
        ((2, 3), (6, 7)),  # EAST  ∥ WEST
    )
    DEFAULT_SPAN_EDGES = (
        (0, 1),  # NORTH span
        (2, 3),  # EAST  span
        (4, 5),  # SOUTH span
        (6, 7),  # WEST  span
    )

    def __init__(self,
                 perp_weight: float = 0.03,
                 parallel_weight: float = 0.03,
                 span_weight: float = 0.0,
                 span_tolerance: float = 0.01,
                 absent_weight: float = 0.0,
                 tolerance_deg: float = 10.0,
                 softmax_temp: float = 10.0,
                 warmup_steps: int = 0,
                 perpendicular_pairs: Optional[Sequence[
                     Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
                 parallel_pairs: Optional[Sequence[
                     Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
                 span_edges: Optional[Sequence[Tuple[int, int]]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.perp_weight = perp_weight
        self.parallel_weight = parallel_weight
        self.span_weight = span_weight
        self.span_tolerance = span_tolerance
        self.absent_weight = absent_weight
        self.tolerance_deg = tolerance_deg
        self.softmax_temp = softmax_temp
        self.warmup_steps = warmup_steps
        # Step counter that auto-advances on every forward() call. Used to
        # linearly ramp geometric weights from 0 -> target over `warmup_steps`
        # gradient steps so the model can first learn to make peaked heatmaps.
        self.register_buffer('_step', torch.zeros(1, dtype=torch.long),
                             persistent=False)

        self.perpendicular_pairs = (perpendicular_pairs
                                    if perpendicular_pairs is not None
                                    else self.DEFAULT_PERPENDICULAR_PAIRS)
        self.parallel_pairs = (parallel_pairs
                               if parallel_pairs is not None
                               else self.DEFAULT_PARALLEL_PAIRS)
        self.span_edges = (span_edges
                           if span_edges is not None
                           else self.DEFAULT_SPAN_EDGES)

        # Pre-compute cosine thresholds for the free-zone clamps
        # For perpendicular: |cos(angle)| should be small. The free zone is
        # angle in [90-tolerance, 90+tolerance], i.e. |cos| <= sin(tolerance).
        self._perp_threshold = abs(torch.sin(
            torch.tensor(radians(tolerance_deg))).item())
        # For parallel: |cos(angle)| should be ~1. The free zone is
        # angle in [-tolerance, tolerance] modulo 180°, i.e. |cos| >= cos(tolerance).
        self._parallel_threshold = cos(radians(tolerance_deg))

    # --------------------------------------------------------------------- helpers
    def _soft_argmax(self, heatmap: Tensor) -> Tensor:
        """Differentiable expected (x, y) coordinate per heatmap channel.

        Returns:
            coords: (B, K, 2) in heatmap-pixel units.
        """
        b, k, h, w = heatmap.shape
        soft = F.softmax(
            heatmap.reshape(b, k, -1) * self.softmax_temp,
            dim=-1
        ).reshape(b, k, h, w)

        device, dtype = heatmap.device, heatmap.dtype
        x_grid = torch.arange(w, dtype=dtype, device=device).view(1, 1, 1, w)
        y_grid = torch.arange(h, dtype=dtype, device=device).view(1, 1, h, 1)

        x = (soft * x_grid).sum(dim=(2, 3))  # (B, K)
        y = (soft * y_grid).sum(dim=(2, 3))  # (B, K)
        return torch.stack([x, y], dim=-1)   # (B, K, 2)

    @staticmethod
    def _per_keypoint_visibility(target_weights: Optional[Tensor]) -> Optional[Tensor]:
        """Reduce target_weights to per-keypoint visibility [B, K]."""
        if target_weights is None:
            return None
        if target_weights.ndim == 2:
            return target_weights
        if target_weights.ndim == 4:
            # collapse the spatial dims so any positive weight counts as visible
            return target_weights.amax(dim=(2, 3))
        raise ValueError(
            f'target_weights must have ndim 2 or 4, got {target_weights.ndim}')

    def _edge_vec_and_vis(self, coords, vis, edge):
        a, b = edge
        v = F.normalize(coords[:, b] - coords[:, a], dim=-1, eps=1e-6)  # (B, 2)
        if vis is None:
            return v, None
        return v, vis[:, a] * vis[:, b]                                 # (B,)

    @staticmethod
    def _edge_len_and_vis(coords, vis, edge):
        """Unnormalised edge LENGTH -- the quantity _edge_vec_and_vis discards."""
        a, b = edge
        length = torch.linalg.vector_norm(coords[:, b] - coords[:, a], dim=-1)
        if vis is None:
            return length, None
        return length, vis[:, a] * vis[:, b]                            # (B,)

    def _reduce_with_vis(self, term: Tensor, vis_mask: Optional[Tensor]) -> Tensor:
        """Average `term` (B,) over visible samples; falls back to plain mean."""
        if vis_mask is None:
            return term.mean()
        denom = vis_mask.sum().clamp(min=1e-6)
        return (term * vis_mask).sum() / denom

    # --------------------------------------------------------------------- forward
    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        # Visibility for the GEOMETRIC terms must come from the ORIGINAL
        # weights: an absent keypoint has no meaningful coordinate, so it must
        # never enter the perpendicular / parallel / span computations.
        vis = self._per_keypoint_visibility(target_weights)

        # Absent-keypoint suppression.
        #
        # The codec already emits an ALL-ZERO target heatmap for a v=0 keypoint
        # -- it is the accompanying weight of 0 that throws that supervision
        # away, leaving the channel with exactly zero gradient. The model is
        # therefore never told that an absent corner means "output nothing",
        # and it learns to peak on whatever looks corner-like: measured on
        # ChewyCrops, 169 corners that do not exist in the image still peak at
        # a mean of 0.83, with 70% above 0.8 -- indistinguishable from the 0.96
        # of real detections, so no confidence gate can separate them.
        #
        # Restoring a nonzero weight turns that existing zero target into
        # suppression. Note this needs no new magic constant: absent_weight=1.0
        # simply means "train this channel like any other", and the target it
        # is trained against is the codec's own output.
        mse_weights = target_weights
        if self.absent_weight > 0 and target_weights is not None:
            absent = (target_weights == 0)
            if absent.any():
                mse_weights = target_weights.clone()
                mse_weights[absent] = self.absent_weight

        # Standard heatmap MSE — always runs at full weight
        mse_loss = super().forward(output, target, mse_weights, mask)

        # Warmup ramp on the geometric weights. Returns 0 → 1 over warmup_steps,
        # then stays at 1. Lets the model first learn to make peaked heatmaps
        # before the soft-argmax-based geometric terms become meaningful.
        if self.warmup_steps > 0 and self.training:
            ramp = float(self._step.item()) / float(self.warmup_steps)
            ramp = min(max(ramp, 0.0), 1.0)
            self._step += 1
        else:
            ramp = 1.0

        if ramp <= 0.0:
            return mse_loss

        coords = self._soft_argmax(output)                  # (B, K, 2)
        # `vis` was computed from the ORIGINAL target_weights above, so absent
        # keypoints stay excluded from every geometric term.

        # Perpendicularity: |cos(angle)| should be ≤ sin(tolerance) --> no penalty
        perp_loss = 0.0
        for e1, e2 in self.perpendicular_pairs:
            v1, vis1 = self._edge_vec_and_vis(coords, vis, e1)
            v2, vis2 = self._edge_vec_and_vis(coords, vis, e2)
            cos_a = (v1 * v2).sum(dim=-1)                    # (B,)
            term = torch.clamp(cos_a.abs() - self._perp_threshold,
                               min=0.0) ** 2                  # (B,)
            edge_vis = (vis1 * vis2) if vis is not None else None
            perp_loss = perp_loss + self._reduce_with_vis(term, edge_vis)
        perp_loss = perp_loss / len(self.perpendicular_pairs)

        # Parallelism: |cos(angle)| should be ≥ cos(tolerance) --> no penalty
        parallel_loss = 0.0
        for e1, e2 in self.parallel_pairs:
            v1, vis1 = self._edge_vec_and_vis(coords, vis, e1)
            v2, vis2 = self._edge_vec_and_vis(coords, vis, e2)
            cos_a = (v1 * v2).sum(dim=-1)                    # (B,)
            term = torch.clamp(self._parallel_threshold - cos_a.abs(),
                               min=0.0) ** 2                  # (B,)
            edge_vis = (vis1 * vis2) if vis is not None else None
            parallel_loss = parallel_loss + self._reduce_with_vis(term, edge_vis)
        parallel_loss = parallel_loss / len(self.parallel_pairs)

        # Span: predicted flap LENGTH should match the labelled length.
        #
        # Both terms above normalise their edge vectors, so they constrain edge
        # DIRECTION and are completely blind to edge LENGTH. A flap that is
        # perfectly parallel to its opposite and perpendicular to its neighbours
        # but 30px too short costs exactly zero. Measured on the OOD set, that
        # shows up as error running ALONG the edges (1.39x the across-edge
        # component; 2.48x at SOUTH_1) and flap spans reading too SHORT on 80%
        # of NORTH/SOUTH edges. Downstream measures corner-to-corner span, so
        # that bias lands straight in the product measurement.
        #
        # The target length is read from the GT heatmap with the SAME
        # soft-argmax used on the prediction. That matters: soft-argmax over a
        # full heatmap pulls slightly toward the centre, so using an exact
        # centroid for the target and soft-argmax for the prediction would make
        # correct predictions look short and push the model to overshoot. Same
        # operator on both sides cancels the bias to first order.
        #
        # NOT an opposite-edge equality constraint (|NORTH| == |SOUTH|): under
        # perspective a nearer flap genuinely images longer than the far one.
        # Measured in the labels, opposite-edge ratios reach 1.18 at p90 and
        # 1.59 at max, so equality would be wrong by more than the bias it aims
        # to fix. Comparing against the per-image label keeps perspective in the
        # target where it belongs.
        span_loss = 0.0
        if self.span_weight > 0:
            with torch.no_grad():
                gt_coords = self._soft_argmax(target)
            for edge in self.span_edges:
                len_pred, edge_vis = self._edge_len_and_vis(coords, vis, edge)
                len_gt, _ = self._edge_len_and_vis(gt_coords, vis, edge)
                rel_err = (len_pred - len_gt).abs() / len_gt.clamp(min=1e-3)
                # Free zone, matching the design of the two angular terms: no
                # penalty inside `span_tolerance` so the term does not chase
                # annotation noise.
                term = torch.clamp(rel_err - self.span_tolerance, min=0.0) ** 2
                span_loss = span_loss + self._reduce_with_vis(term, edge_vis)
            span_loss = span_loss / len(self.span_edges)

        return (mse_loss
                + ramp * self.perp_weight * perp_loss
                + ramp * self.parallel_weight * parallel_loss
                + ramp * self.span_weight * span_loss)
