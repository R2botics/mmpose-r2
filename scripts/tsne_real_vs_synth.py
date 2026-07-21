"""t-SNE visualization of real vs synthetic samples in the model's feature space.

Extracts backbone features from a trained model for both real and synthetic
images, projects them into 2D with t-SNE, and plots them colored by source.

Interpretation:
    - CLEARLY SEPARATED clusters -> the model has learned to distinguish the
      two distributions. Big domain gap. Combined training will pull the model
      toward whichever domain dominates the batch (which is what hurt us).
    - OVERLAPPING clusters -> the two distributions look similar in feature
      space. Combined training should work well.
    - MIXED (partial overlap) -> some samples are close, others are outliers.
      The outliers are the ones causing training divergence.

Also computes a simple linear separability score: fits a logistic regression
to the extracted features and reports classification accuracy. Near 50% is
ideal (indistinguishable). Near 100% means very separable.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

# Register custom modules so init_model can build the network
import mmpose.datasets.transforms.rgbd            # noqa: F401
import mmpose.models.data_preprocessors.nchannel  # noqa: F401
import mmpose.models.losses.geometric_loss        # noqa: F401

from mmpose.apis import init_model


def load_and_preprocess(color_path, depth_path, mean, std, input_size=256):
    rgb = cv2.imread(str(color_path))
    if rgb is None:
        return None
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (input_size, input_size))

    if depth_path is not None and Path(depth_path).exists():
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (input_size, input_size),
                           interpolation=cv2.INTER_NEAREST)
        mask = (depth > 0).astype(np.uint8) * 255
        x = np.dstack([rgb,
                       depth.astype(np.float32)[..., None].squeeze(),
                       mask]).astype(np.float32)
        # dstack above collapses depth so redo cleanly
        x = np.zeros((*rgb.shape[:2], 5), dtype=np.float32)
        x[..., :3] = rgb.astype(np.float32)
        x[..., 3] = depth.astype(np.float32)
        x[..., 4] = mask.astype(np.float32)
    else:
        x = rgb.astype(np.float32)

    if len(mean) != x.shape[2]:
        raise ValueError(
            f'Mean/std has {len(mean)} channels but input has {x.shape[2]}. '
            f'Check config/data compatibility.')
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)[None]  # NHWC -> (1, C, H, W)
    return torch.from_numpy(x).float()


def extract_backbone_feat(model, x):
    """Extract global-avg-pooled features from the backbone's highest-res branch."""
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        feats = model.backbone(x)
    # HRNet backbone returns a tuple: (highest-res, ..., lowest-res)
    # We use the highest-res (branch 0) since it's what feeds the pose head.
    top = feats[0] if isinstance(feats, (tuple, list)) else feats
    pooled = top.mean(dim=[2, 3])  # (1, C)
    return pooled.cpu().numpy().flatten()


def mask_channels(x: torch.Tensor, keep_channels: list) -> torch.Tensor:
    """Return a copy of `x` where channels NOT in `keep_channels` are zeroed.

    Because the input has already been normalized (val - mean) / std, zero in
    the normalized space corresponds to the channel's mean value in the
    original data — a "neutral" input the model doesn't take as signal.
    """
    x = x.clone()
    n_channels = x.shape[1]
    zero_channels = [c for c in range(n_channels) if c not in keep_channels]
    if zero_channels:
        x[:, zero_channels] = 0.0
    return x


def collect_features(model, image_dir, depth_dir, mean, std, max_samples,
                     label, keep_channels=None):
    """Extract features. If `keep_channels` is None, all channels flow through.
    Otherwise, other channels are zeroed after normalization."""
    color_paths = sorted(Path(image_dir).glob('*_cropped_color.png'))[:max_samples]
    features = []
    for cp in color_paths:
        dp = Path(depth_dir) / cp.name.replace('_cropped_color.png',
                                               '_aligned_depth.png')
        x = load_and_preprocess(cp, dp, mean, std)
        if x is None:
            continue
        if keep_channels is not None:
            x = mask_channels(x, keep_channels)
        f = extract_backbone_feat(model, x)
        features.append(f)
    print(f'  {label}: extracted {len(features)} feature vectors '
          f'(dim {features[0].size if features else 0})')
    return np.array(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/hrnet-w32_8-kp_udp_rgbd_geom.py')
    parser.add_argument('--checkpoint',
                        default='work_dirs/hrnet_udp_rgbd_geom_test_01/best_coco_AP_epoch_30.pth')
    parser.add_argument('--real-dir', default='data/RSC_Keypoints_RGBD')
    parser.add_argument('--synth-dir',
                        default='/home/siddharth/Deps/r2-synthbox/out/rgbd_crops')
    parser.add_argument('--output', default='work_dirs/tsne_real_vs_synth.png')
    parser.add_argument('--max-real', type=int, default=10000,
                        help='Cap on real samples to extract (default: use all)')
    parser.add_argument('--max-synth', type=int, default=10000,
                        help='Cap on synth samples to extract (default: use all)')
    parser.add_argument('--perplexity', type=float, default=30)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ablate', action='store_true',
                        help='Run modality ablation: 3 t-SNE plots for '
                             '[all channels, RGB only, depth+mask only]')
    args = parser.parse_args()

    init_default_scope('mmpose')

    print(f'Loading model: {args.checkpoint}')
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    model.eval()

    mean = np.array(cfg.model.data_preprocessor.mean, dtype=np.float32)
    std = np.array(cfg.model.data_preprocessor.std, dtype=np.float32)
    print(f'Using {len(mean)}-channel normalization from config')
    print()

    from sklearn.manifold import TSNE
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def separability_verdict(sep_acc):
        if sep_acc < 0.60:
            return 'INDISTINGUISHABLE'
        elif sep_acc < 0.80:
            return 'PARTIALLY SEPARABLE'
        elif sep_acc < 0.95:
            return 'MOSTLY SEPARABLE'
        return 'HIGHLY SEPARABLE'

    # Build the list of experiments to run
    n_channels = len(mean)
    experiments = [('all channels', None)]
    if args.ablate and n_channels >= 4:
        # Channel layout (RGBD training): 0=R, 1=G, 2=B, 3=depth, 4=mask
        experiments.append(('RGB only', [0, 1, 2]))
        experiments.append(('depth + mask only', [3, 4] if n_channels == 5 else [3]))

    n_experiments = len(experiments)
    fig, axes = plt.subplots(1, n_experiments,
                             figsize=(9 * n_experiments, 8),
                             squeeze=False)
    axes = axes[0]

    results = []
    for i, (name, keep_channels) in enumerate(experiments):
        print()
        print(f'=== Experiment {i+1}/{n_experiments}: {name} ===')
        print(f'Extracting features (keep_channels={keep_channels})...')
        real_feats = collect_features(
            model,
            Path(args.real_dir) / 'images',
            Path(args.real_dir) / 'depth',
            mean, std, args.max_real, 'Real', keep_channels)
        synth_feats = collect_features(
            model,
            Path(args.synth_dir) / 'images',
            Path(args.synth_dir) / 'depth',
            mean, std, args.max_synth, 'Synthetic', keep_channels)

        features = np.vstack([real_feats, synth_feats])
        labels = np.concatenate([np.zeros(len(real_feats)),
                                 np.ones(len(synth_feats))])

        embedded = TSNE(
            n_components=2, perplexity=args.perplexity,
            random_state=42, init='pca',
            learning_rate='auto').fit_transform(features)

        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, features, labels, cv=5)
        sep_acc = scores.mean()
        verdict = separability_verdict(sep_acc)
        print(f'Linear separability: {sep_acc:.3f}   verdict: {verdict}')
        results.append((name, sep_acc, verdict))

        real_mask = labels == 0
        ax = axes[i]
        ax.scatter(embedded[real_mask, 0], embedded[real_mask, 1],
                   alpha=0.6, s=30, label=f'Real (n={real_mask.sum()})',
                   c='#1f77b4', edgecolors='navy', linewidth=0.5)
        ax.scatter(embedded[~real_mask, 0], embedded[~real_mask, 1],
                   alpha=0.6, s=30, label=f'Synthetic (n={(~real_mask).sum()})',
                   c='#ff7f0e', edgecolors='darkred', linewidth=0.5)
        ax.set_title(f'{name}\nSeparability: {sep_acc:.1%} — {verdict}',
                     fontsize=11)
        ax.set_xlabel('t-SNE dim 1')
        ax.set_ylabel('t-SNE dim 2')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(alpha=0.3)

    plt.suptitle(
        'Domain shift ablation: which modality is causing the separation?',
        fontsize=13, y=1.02)
    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print()
    print(f'Saved plot to: {out}')
    print()
    print('Summary:')
    for name, acc, verdict in results:
        print(f'  {name:<20s}  {acc:.1%}   {verdict}')


if __name__ == '__main__':
    main()
