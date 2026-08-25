"""Transforms for RGBD pose estimation.

Loads RGB image + aligned 16-bit depth, producing 3, 4 or 5 channels:
    [R, G, B]                                   channels=3  (depth ablation)
    [R, G, B, depth(uint16)]                    channels=4
    [R, G, B, depth(uint16), mask(0 or 255)]    channels=5  (default)

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
    """Load RGB + aligned depth, stack into a 3/4/5-channel image.

    Required keys:
        - img_path  (path to the color image, ending in `_cropped_color.png`)

    Modified keys:
        - img       (HxWx{3,4,5} float32, order R, G, B[, depth[, mask]])
        - img_shape
        - ori_shape

    Args:
        color_suffix:  suffix used in color filenames
        depth_suffix:  suffix used in depth filenames
        color_dir:     directory name containing color images
        depth_dir:     directory name containing depth images
        channels:      how many channels to emit. 5 = [R,G,B,depth,mask]
            (default, unchanged behaviour); 4 = [R,G,B,depth]; 3 = [R,G,B],
            and the depth file is not even opened.

            This exists so the depth ablation is a ONE-LINE config change that
            keeps every other stage of the pipeline byte-identical -- same
            crop, same affine, same augmentation RNG draw order. Swapping in
            mmpose's stock `LoadImage` for the RGB arm would also change which
            reader decodes the PNG and how the bbox is derived, and then a
            difference in the numbers is no longer attributable to depth.

            Note what channel 4 is FOR. Channel 3 is height above the
            conveyor, and a zero there is three different things: background
            outside the box, box contents below the height cutoff, and genuine
            sensor dropout. Channel 4 does not disambiguate them either -- it
            is exactly `depth > 0`, so it carries no information channel 3
            does not already contain. What it buys is REPRESENTATION: a
            network reading channel 3 alone has to spend capacity learning
            that one particular value of a continuous input means "no
            measurement", which a linear patch projection cannot express at
            all. Channel 4 hands it that indicator directly. Whether that is
            worth a channel is exactly what the 4-vs-5 ablation measures.
    """

    VALID_CHANNELS = (3, 4, 5)

    def __init__(self,
                 color_suffix: str = '_cropped_color.png',
                 depth_suffix: str = '_aligned_depth.png',
                 color_dir: str = 'images',
                 depth_dir: str = 'depth',
                 channels: int = 5):
        if channels not in self.VALID_CHANNELS:
            raise ValueError(
                f'channels must be one of {self.VALID_CHANNELS}, got {channels}')
        self.color_suffix = color_suffix
        self.depth_suffix = depth_suffix
        self.color_dir = color_dir
        self.depth_dir = depth_dir
        self.channels = channels

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

        if self.channels == 3:
            img = np.ascontiguousarray(rgb.astype(np.float32))
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]
            return results

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

        planes = [rgb.astype(np.float32), depth.astype(np.float32)]
        if self.channels == 5:
            # Validity mask from depth holes (0 = hole, 255 = real depth)
            planes.append(((depth > 0).astype(np.uint8) * 255).astype(np.float32))

        rgbdm = np.dstack(planes)

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
