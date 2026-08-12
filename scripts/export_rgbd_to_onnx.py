"""Export the 5-channel RGBD HRNet pose model to ONNX.

The exported graph takes a single 5-channel tensor (already preprocessed —
normalized, in NCHW order with channels R, G, B, depth, mask) and returns
the 8-keypoint heatmap tensor of shape (B, 8, 64, 64).

Heatmap → keypoint coordinate decoding (the UDPHeatmap codec's `decode()`
step) is intentionally NOT included in the ONNX graph. It uses numpy ops
that don't trace cleanly. Do that decoding step in your deployment runtime
on the heatmap output (it's just an argmax + sub-pixel offset; trivial).

Usage:
    python scripts/export_rgbd_to_onnx.py \\
        --config configs/hrnet-w32_8-kp_udp_rgbd_geom.py \\
        --checkpoint work_dirs/hrnet_udp_rgbd_geom_test_01/best_coco_AP_epoch_30.pth \\
        --output work_dirs/exported/hrnet_rgbd_geom.onnx
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.registry import init_default_scope

# Register everything custom so init_model can build the network
import mmpose.datasets.transforms.rgbd            # noqa: F401
import mmpose.models.data_preprocessors.nchannel  # noqa: F401
import mmpose.models.losses.geometric_loss        # noqa: F401

from mmpose.apis import init_model


class HeatmapOnlyWrapper(nn.Module):
    """Wrap a TopdownPoseEstimator so its forward is just:
        5-channel preprocessed tensor -> heatmaps tensor.

    This skips the data preprocessor (do that externally before inference)
    and the head's decode step (do that externally on the heatmap).
    """

    def __init__(self, pose_estimator):
        super().__init__()
        self.backbone = pose_estimator.backbone
        self.head = pose_estimator.head
        # neck is optional; HRNet pose configs don't use one
        self.neck = getattr(pose_estimator, 'neck', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)

        # HeatmapHead.forward returns predicted heatmaps directly when given
        # the backbone features (no decode step happens here).
        return self.head.forward(feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/hrnet-w32_8-kp_udp_rgbd_geom.py',
                        help='Path to mmpose config (default: RGBD+Geom)')
    parser.add_argument('--checkpoint',
                        default='work_dirs/hrnet_udp_rgbd_geom_test_01/best_coco_AP_epoch_30.pth',
                        help='Path to trained checkpoint')
    parser.add_argument('--output',
                        default='work_dirs/exported/hrnet_rgbd.onnx')
    parser.add_argument('--input-size', type=int, nargs=2, default=[256, 256],
                        help='(H, W) of the input tensor')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Fixed batch size baked into the ONNX graph '
                             '(ignored if --dynamic-batch is used)')
    parser.add_argument('--num-channels', type=int, default=5,
                        help='Input channels (5 for RGBD+mask, 4 for RGBD, 3 for RGB-only)')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version')
    parser.add_argument('--dynamic-batch', action='store_true',
                        help='Make batch dimension dynamic in the exported graph')
    parser.add_argument('--device', default='cpu',
                        help='Use cpu for export — guarantees no CUDA-only ops slip in')
    args = parser.parse_args()

    # Fail fast and legibly. torch.onnx.export imports `onnx` internally
    # (onnx_proto_utils._add_onnxscript_fn), so a missing onnx surfaces as an
    # OnnxExporterError from deep inside torch *after* the checkpoint has been
    # loaded and the sanity forward has run — which reads like an export bug
    # rather than a missing dependency.
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise SystemExit(
            'onnx is required to export (torch.onnx.export imports it '
            'internally), but it is not installed.\n'
            '    pip install onnx onnxruntime "numpy<2"\n'
            'Keep the numpy pin: an unconstrained install pulls numpy 2.x, '
            'which breaks the mmcv and xtcocotools C extensions.')

    init_default_scope('mmpose')

    print(f'Loading config: {args.config}')
    cfg = Config.fromfile(args.config)

    print(f'Loading checkpoint: {args.checkpoint}')
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    # Wrap so we have a clean tensor-in, tensor-out signature
    wrapped = HeatmapOnlyWrapper(model).to(args.device).eval()

    # Verify with a dummy forward
    h, w = args.input_size
    dummy = torch.randn(args.batch_size, args.num_channels, h, w,
                        device=args.device)
    with torch.no_grad():
        out = wrapped(dummy)
    print(f'Sanity check - input: {tuple(dummy.shape)} -> heatmaps: {tuple(out.shape)}')

    # Export
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch'},
            'heatmaps': {0: 'batch'},
        }

    print(f'Exporting to {out_path}')
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            str(out_path),
            input_names=['input'],
            output_names=['heatmaps'],
            opset_version=args.opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    # Sanity check the exported ONNX
    try:
        import onnx
        loaded = onnx.load(str(out_path))
        onnx.checker.check_model(loaded)
        print(f'ONNX model validated. File size: {out_path.stat().st_size / 1e6:.1f} MB')
    except ImportError:
        print('onnx library not installed — skipping graph validation')

    # Optional: run inference to confirm parity with PyTorch
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_path),
                                    providers=['CPUExecutionProvider'])
        ort_out = sess.run(['heatmaps'], {'input': dummy.cpu().numpy()})[0]
        ort_tensor = torch.from_numpy(ort_out)
        max_diff = (out.cpu() - ort_tensor).abs().max().item()
        print(f'PyTorch vs ONNX max abs diff: {max_diff:.6f} (should be < 1e-4)')
    except ImportError:
        print('onnxruntime not installed — skipping parity check')
        print('  pip install onnxruntime  # to enable')

    print()
    print(f'Done. ONNX model: {out_path}')
    print(f'  Input:  (B, {args.num_channels}, {h}, {w}) float32 — already normalized')
    print(f'  Output: (B, 8, {h // 4}, {w // 4}) float32 — heatmaps')
    print(f'  Decode heatmaps -> (x, y) externally with argmax + 0.25 stride')


if __name__ == '__main__':
    main()
