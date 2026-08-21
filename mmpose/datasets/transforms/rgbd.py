"""Transforms for RGBD pose estimation.

Loads RGB image + aligned 16-bit depth, produces a 5-channel input:
    [R, G, B, depth(uint16), mask(0 or 255)]

The mask is generated on the fly from `depth > 0`, acting as a trust signal
that the depth value at that pixel is real (vs filtered background or sensor hole).
"""
import os.path as osp

import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS

from .common_transforms import RandomFlip


@TRANSFORMS.register_module()
class LoadRGBDImage(BaseTransform):
    """Load RGB + aligned depth, stack into 5-channel image.

    Required keys:
        - img_path  (path to the color image, ending in `_cropped_color.png`)

    Modified keys:
        - img       (HxWx5 float32, channel order R, G, B, depth, mask)
        - img_shape
        - ori_shape

    Args:
        color_suffix:  suffix used in color filenames
        depth_suffix:  suffix used in depth filenames
        color_dir:     directory name containing color images
        depth_dir:     directory name containing depth images
    """

    def __init__(self,
                 color_suffix: str = '_cropped_color.png',
                 depth_suffix: str = '_aligned_depth.png',
                 color_dir: str = 'images',
                 depth_dir: str = 'depth'):
        self.color_suffix = color_suffix
        self.depth_suffix = depth_suffix
        self.color_dir = color_dir
        self.depth_dir = depth_dir

    def transform(self, results: dict) -> dict:
        rgb_path = results['img_path']
        depth_path = (rgb_path
                      .replace(f'/{self.color_dir}/', f'/{self.depth_dir}/')
                      .replace(self.color_suffix, self.depth_suffix))

        # Load RGB (channel_order='rgb' so we don't need bgr_to_rgb later)
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise FileNotFoundError(f'Could not read color image: {rgb_path}')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # HxWx3 uint8

        # Load depth (uint16, single channel)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f'Could not read depth image: {depth_path}')
        if depth.ndim != 2:
            raise ValueError(
                f'Expected single-channel depth, got shape {depth.shape}')

        if depth.shape != rgb.shape[:2]:
            raise ValueError(
                f'Depth shape {depth.shape} != color shape {rgb.shape[:2]} '
                f'for {rgb_path}. Check your alignment.')

        # Validity mask from depth holes (0 = hole, 255 = real depth)
        mask = (depth > 0).astype(np.uint8) * 255

        # Stack to 5-channel float32:  R, G, B, depth, mask
        rgbdm = np.dstack([
            rgb.astype(np.float32),
            depth.astype(np.float32),
            mask.astype(np.float32),
        ])

        results['img'] = rgbdm
        results['img_shape'] = rgbdm.shape[:2]
        results['ori_shape'] = rgbdm.shape[:2]
        return results


@TRANSFORMS.register_module()
class PhotometricDistortionRGBOnly(BaseTransform):
    """Apply PhotometricDistortion to the RGB channels only.

    The depth and mask channels (3, 4) are preserved unchanged.
    Wraps `mmpose.datasets.transforms.PhotometricDistortion` over a temporary
    3-channel view.
    """

    def __init__(self, **kwargs):
        # Import lazily to avoid circular imports
        from mmpose.datasets.transforms import PhotometricDistortion
        self.photo = PhotometricDistortion(**kwargs)

    def transform(self, results: dict) -> dict:
        img = results['img']
        if img.shape[2] < 4:
            return self.photo.transform(results)

        rgb = img[..., :3].copy()
        extra = img[..., 3:].copy()  # depth + mask (and anything else later)

        # Apply photometric distortion to the RGB-only image
        photo_results = dict(results)
        photo_results['img'] = rgb
        photo_results = self.photo.transform(photo_results)

        # Recombine
        results['img'] = np.concatenate([photo_results['img'], extra], axis=2)
        return results


@TRANSFORMS.register_module()
class RandomFlipVertical(RandomFlip):
    """Top-bottom mirror with the correct keypoint permutation.

    Plain ``RandomFlip(direction='vertical')`` is WRONG for this dataset.
    ``flip_keypoints()`` applies ``results['flip_indices']`` for both
    directions, and ``flip_indices`` is built from the ``swap=`` fields in the
    dataset metainfo, which describe the HORIZONTAL mirror. Using it for a
    vertical flip permutes keypoints into the wrong channels and raises no
    error -- the run just trains on corrupted labels.

    The vertical permutation differs because a top-bottom mirror moves NORTH
    onto SOUTH, while a left-right mirror keeps NORTH at the top:

        horizontal:  N1<->N2  E1<->W2  E2<->W1  S1<->S2   (metainfo `swap=`)
        vertical:    N1<->S2  N2<->S1  E1<->E2  W1<->W2   (this transform)

    Derived from the mean keypoint layout in the box's own frame: each physical
    box corner carries one N/S-flap keypoint and one E/W-flap keypoint, and a
    vertical mirror maps a corner to the one below/above it while keeping each
    keypoint within its own flap family.

    Why bother: the Chewy crops label ``E2/S1/S2/W1`` on only ~half as many
    boxes as ``N1/N2/E1/W2`` (those corners fall outside the crop), leaving a
    19%-of-mean spread in per-channel supervision. Horizontal flip cannot fix
    it -- it pairs (E2,W1) and (S1,S2), which are BOTH starved. Vertical flip
    pairs each starved channel with an abundant one and cuts the spread to 3%.

    Composes with the horizontal flip: applying both gives the full
    {identity, H, V, 180-degree rotation} group.

    Args:
        prob: Probability of flipping. Defaults to 0.5.
        flip_indices: Override the vertical permutation. Defaults to the
            8-keypoint box layout above.
    """

    #: N1<->S2, N2<->S1, E1<->E2, W1<->W2
    DEFAULT_VERTICAL_FLIP_INDICES = [5, 4, 3, 2, 1, 0, 7, 6]

    def __init__(self, prob=0.5, flip_indices=None):
        super().__init__(prob=prob, direction='vertical')
        self.vertical_flip_indices = (
            list(flip_indices) if flip_indices is not None
            else list(self.DEFAULT_VERTICAL_FLIP_INDICES))

    def transform(self, results: dict) -> dict:
        # Substitute the vertical permutation for the duration of the call,
        # then restore. `flip_indices` is carried into metainfo by
        # PackPoseInputs and reused by flip_test at inference, where the
        # HORIZONTAL indices are the correct ones -- so leaving ours behind
        # would silently break test-time flip augmentation.
        saved_indices = results.get('flip_indices')
        # A horizontal RandomFlip earlier in the pipeline may already have set
        # these; super().transform() overwrites them unconditionally, so keep
        # the record that a flip happened.
        was_flipped = results.get('flip', False)
        prev_direction = results.get('flip_direction')

        results['flip_indices'] = self.vertical_flip_indices
        try:
            results = super().transform(results)
        finally:
            if saved_indices is not None:
                results['flip_indices'] = saved_indices
            else:
                results.pop('flip_indices', None)

        if was_flipped and not results.get('flip', False):
            results['flip'] = True
            results['flip_direction'] = prev_direction
        return results
