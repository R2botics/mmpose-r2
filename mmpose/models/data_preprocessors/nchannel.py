"""N-channel pose data preprocessor.

Subclasses `PoseDataPreprocessor` to support arbitrary channel counts (e.g. 5 for
RGBD + mask). The parent's `ImgDataPreprocessor.__init__` enforces a 1/3-channel
assertion on `mean`/`std`; we bypass that here.
"""
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from mmengine.utils import is_seq_of

from mmpose.registry import MODELS
from .data_preprocessor import PoseDataPreprocessor


@MODELS.register_module()
class NChannelPoseDataPreprocessor(PoseDataPreprocessor):
    """PoseDataPreprocessor that accepts arbitrary input channel counts.

    Args mirror PoseDataPreprocessor, but `mean` and `std` can have any length
    (typically matching the number of input channels). The bgr_to_rgb /
    rgb_to_bgr swaps don't make sense for non-3-channel inputs and are not
    supported here — convert channel order in the LoadImage transform instead.
    """

    def __init__(self,
                 mean: Optional[Sequence[float]] = None,
                 std: Optional[Sequence[float]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None):
        # Bypass ImgDataPreprocessor.__init__ (which has the 1/3-channel
        # assertion) and the PoseDataPreprocessor.__init__ that calls it.
        # We go straight to BaseDataPreprocessor.__init__ via nn.Module.
        nn.Module.__init__(self)
        self._non_blocking = non_blocking
        self._device = torch.device('cpu')

        # ImgDataPreprocessor fields
        self._channel_conversion = False  # handle channel order in LoadImage
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

        if mean is not None:
            assert std is not None, ('`std` must be set when `mean` is set')
            assert is_seq_of(mean, (int, float)) and is_seq_of(std, (int, float)), \
                '`mean` and `std` must be sequences of int/float'
            assert len(mean) == len(std), (
                f'mean and std must have the same length, '
                f'got mean={len(mean)} std={len(std)}')

            self._enable_normalize = True
            self.register_buffer(
                'mean',
                torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1),
                False)
            self.register_buffer(
                'std',
                torch.tensor(std, dtype=torch.float32).view(-1, 1, 1),
                False)
        else:
            self._enable_normalize = False

        # PoseDataPreprocessor fields
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None
