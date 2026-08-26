"""Class-agnostic corner heatmap: "at most 8 keypoints", no cardinal identity.

WHY THIS EXISTS
    The production codec gives each flap corner a NAME -- NORTH_1 .. WEST_2 --
    and dedicates one heatmap channel to each. The model therefore has to solve
    two problems at once: WHERE the corners are, and WHICH corner each one is.
    This codec drops the second. All eight corners are rendered into ONE
    channel, and decoding pulls the peaks back out as an unordered set.

    It exists to answer one question: how much localisation accuracy is the
    naming costing? Score it with AgnosticKeypointDistanceMetric, which matches
    predictions to ground truth optimally, so a permutation is free.

    NOT a production path. The downstream span measurement needs to know which
    two keypoints mark one physical corner, and this codec cannot say. It is a
    measuring instrument for the cardinal formulation, nothing more.

DIFFERENCES FROM UDPHeatmap, all forced by the single channel
    encode:  the K per-keypoint gaussians are MAX-reduced into one map. Max,
             not sum: two corners closer than a couple of sigma would
             otherwise build a single brighter blob whose peak sits between
             them, inventing a corner where there is none.
    decode:  peaks instead of one-argmax-per-channel. 3x3 dilation for
             non-maximum suppression, top-K above a score floor, then the same
             quadratic sub-pixel refinement UDP applies.

    Fewer than K peaks is a legitimate outcome -- "at most 8". Missing slots
    come back as (0, 0) with score 0 so the array stays (1, K, 2), and the
    metric drops zero-score predictions rather than scoring them as errors.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils import generate_udp_gaussian_heatmaps


@KEYPOINT_CODECS.register_module()
class AgnosticUDPHeatmap(BaseKeypointCodec):
    """Single-channel corner heatmap with set-valued decoding.

    Args:
        input_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        sigma (float): Gaussian sigma in heatmap pixels
        max_keypoints (int): Most peaks to return per image
        score_thr (float): Peaks below this are not returned at all
        nms_kernel (int): Odd window for the local-max suppression
    """

    auxiliary_encode_keys = set()

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 sigma: float = 2.,
                 max_keypoints: int = 8,
                 score_thr: float = 0.1,
                 nms_kernel: int = 5) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.max_keypoints = max_keypoints
        self.score_thr = score_thr
        assert nms_kernel % 2 == 1, '`nms_kernel` must be odd'
        self.nms_kernel = nms_kernel
        self.scale_factor = ((np.array(input_size) - 1) /
                             (np.array(heatmap_size) - 1)).astype(np.float32)

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Render every visible keypoint into a SINGLE heatmap channel."""
        assert keypoints.shape[0] == 1, (
            f'{self.__class__.__name__} only supports single-instance encoding')
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        per_kpt, kpt_weights = generate_udp_gaussian_heatmaps(
            heatmap_size=self.heatmap_size,
            keypoints=keypoints / self.scale_factor,
            keypoints_visible=keypoints_visible,
            sigma=self.sigma)

        # MAX, not sum -- see the module docstring.
        merged = per_kpt.max(axis=0, keepdims=True)

        # One channel, so one weight: supervise the map whenever any corner is
        # labelled. Zero only for a frame with nothing visible at all.
        weight = np.array([float(kpt_weights.max())], dtype=np.float32)
        return dict(heatmaps=merged, keypoint_weights=weight)

    def _peaks(self, hm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sub-pixel peak locations and scores from one HxW map."""
        k = self.nms_kernel
        local_max = cv2.dilate(hm, np.ones((k, k), np.uint8))
        cand = (hm >= local_max) & (hm > self.score_thr)
        ys, xs = np.nonzero(cand)
        if ys.size == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, ), np.float32)

        scores = hm[ys, xs]
        order = np.argsort(-scores)[:self.max_keypoints]
        ys, xs, scores = ys[order], xs[order], scores[order]

        H, W = hm.shape
        coords = np.stack([xs, ys], axis=-1).astype(np.float32)
        # Quadratic sub-pixel refinement, the same shift UDP applies: fit a
        # parabola through the peak and its two neighbours per axis.
        for i, (x, y) in enumerate(zip(xs, ys)):
            if 0 < x < W - 1:
                dx = hm[y, x + 1] - hm[y, x - 1]
                dxx = hm[y, x + 1] + hm[y, x - 1] - 2 * hm[y, x]
                if abs(dxx) > 1e-9:
                    coords[i, 0] -= 0.5 * dx / dxx
            if 0 < y < H - 1:
                dy = hm[y + 1, x] - hm[y - 1, x]
                dyy = hm[y + 1, x] + hm[y - 1, x] - 2 * hm[y, x]
                if abs(dyy) > 1e-9:
                    coords[i, 1] -= 0.5 * dy / dyy
        return coords, scores.astype(np.float32)

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pull an unordered corner set out of the single channel.

        Returns keypoints (1, max_keypoints, 2) and scores (1, max_keypoints).
        Unfilled slots are (0, 0) with score 0 -- "at most 8" means fewer is a
        valid answer, and the metric drops zero-score entries.
        """
        hm = encoded[0].copy()
        coords, scores = self._peaks(hm)

        K = self.max_keypoints
        kpts = np.zeros((K, 2), np.float32)
        scr = np.zeros((K, ), np.float32)
        n = min(len(coords), K)
        if n:
            kpts[:n] = coords[:n] * self.scale_factor
            scr[:n] = scores[:n]
        return kpts[None], scr[None]
