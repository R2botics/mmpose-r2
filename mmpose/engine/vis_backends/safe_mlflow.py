"""MLflow visualization backend with sanitized metric names.

MLflow's REST API rejects metric names containing characters outside:
    alphanumerics, _, -, ., space, :, /

mmpose's CocoMetric produces names like ``coco/AP (M)`` that contain
parentheses, which crash training the first time validation metrics are
logged to a remote MLflow server.

This subclass silently rewrites disallowed characters to underscores before
handing the batch off to MLflow. Use in configs as:

    dict(type='SafeMLflowVisBackend', tracking_uri=..., ...)

instead of the stock ``MLflowVisBackend``.
"""
import re
from typing import Any, Dict, Optional

from mmengine.registry import VISBACKENDS
from mmengine.visualization import MLflowVisBackend


# MLflow allows: A-Z a-z 0-9 _ - . space : /
_INVALID_METRIC_CHARS = re.compile(r'[^\w\-. :/]')


def _sanitize(name: str) -> str:
    return _INVALID_METRIC_CHARS.sub('_', name)


@VISBACKENDS.register_module()
class SafeMLflowVisBackend(MLflowVisBackend):
    """MLflowVisBackend that sanitizes metric names before logging."""

    def add_scalar(self,
                   name: str,
                   value: Any,
                   step: int = 0,
                   **kwargs) -> None:
        super().add_scalar(_sanitize(name), value, step, **kwargs)

    def add_scalars(self,
                    scalar_dict: Dict[str, Any],
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        sanitized = {_sanitize(k): v for k, v in scalar_dict.items()}
        super().add_scalars(sanitized, step, file_path, **kwargs)
