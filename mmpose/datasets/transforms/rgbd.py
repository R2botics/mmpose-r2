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
