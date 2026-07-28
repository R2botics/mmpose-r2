#!/usr/bin/env bash
#
# Create the `mmpose-r2` environment for training and inference on the
# box-flap HRNet models.
#
# Nothing conda-like is assumed to be installed; micromamba is bootstrapped
# into ~/.local/bin if no conda/mamba/micromamba is already on PATH.
#
# Usage:
#     bash scripts/setup_env.sh              # create (skips if env exists)
#     bash scripts/setup_env.sh --force      # delete and recreate
#     bash scripts/setup_env.sh --verify     # only run the import checks
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="mmpose-r2"
ENV_FILE="${REPO_ROOT}/environment.yml"

FORCE=0
VERIFY_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --force)  FORCE=1 ;;
    --verify) VERIFY_ONLY=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# --------------------------------------------------------------- package manager
# Prefer an existing installation; otherwise bootstrap micromamba.
PKG=""
# A previous run may have installed micromamba here without the user having
# added it to their shell rc yet — look before re-downloading it.
if [[ -x "${HOME}/.local/bin/micromamba" ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
for candidate in micromamba mamba conda; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PKG="$candidate"
    break
  fi
done

if [[ -z "$PKG" ]]; then
  echo "==> No conda/mamba/micromamba found. Bootstrapping micromamba."
  command -v curl >/dev/null 2>&1 || { echo "curl is required to bootstrap micromamba" >&2; exit 1; }
  mkdir -p "${HOME}/.local/bin"
  # The tarball contains bin/micromamba; extract just that.
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C "${HOME}/.local" bin/micromamba
  export PATH="${HOME}/.local/bin:${PATH}"
  PKG="micromamba"
  echo "==> micromamba installed at ${HOME}/.local/bin/micromamba"
  echo "    Add this to your ~/.bashrc to keep it on PATH:"
  echo "        export PATH=\"\${HOME}/.local/bin:\${PATH}\""
  echo "        export MAMBA_ROOT_PREFIX=\"\${HOME}/micromamba\""
fi

# micromamba needs a root prefix even when already installed.
if [[ "$PKG" == "micromamba" ]]; then
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
fi

echo "==> Using package manager: $PKG"

# Run a command inside the environment.
run_in_env() {
  "$PKG" run -n "$ENV_NAME" "$@"
}

env_exists() {
  # micromamba keeps envs under $MAMBA_ROOT_PREFIX/envs; the `env list` table
  # format has changed between releases, so check the directory first.
  if [[ "$PKG" == "micromamba" && -d "${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}" ]]; then
    return 0
  fi
  "$PKG" env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"
}

# --------------------------------------------------------------- create env
if [[ "$VERIFY_ONLY" -eq 0 ]]; then
  if env_exists; then
    if [[ "$FORCE" -eq 1 ]]; then
      echo "==> Removing existing '${ENV_NAME}' environment"
      "$PKG" env remove -n "$ENV_NAME" -y
    else
      echo "==> Environment '${ENV_NAME}' already exists; skipping creation."
      echo "    Re-run with --force to recreate it."
    fi
  fi

  if ! env_exists; then
    echo "==> Creating '${ENV_NAME}' from ${ENV_FILE}"
    echo "    (downloads ~3-4 GB: pytorch + cuda runtime + mmcv)"
    # Older `conda env create` rejects -y; mamba/micromamba accept it.
    if [[ "$PKG" == "conda" ]]; then
      "$PKG" env create -f "$ENV_FILE"
    else
      "$PKG" env create -f "$ENV_FILE" -y
    fi
  fi

  # mmpose itself, editable. Done here rather than in environment.yml so the
  # path resolves against the repo, not the caller's working directory.
  # --no-deps: environment.yml already provides everything, and letting pip
  # resolve requirements/runtime.txt would drag in chumpy, which does not
  # build on Python 3.10.
  echo "==> Installing mmpose (editable, --no-deps) from ${REPO_ROOT}"
  run_in_env python -m pip install --no-deps -e "$REPO_ROOT"

  # ------------------------------------------------------------- .mim tree
  # setup.py's add_mim_extension() only fires on the legacy
  # `python setup.py develop` path (setup.py:134 tests for 'develop' in
  # sys.argv). Modern pip performs a PEP 660 editable build, so 'develop'
  # never appears and mmpose/.mim is silently never created. Every config
  # here inherits `mmpose::_base_/default_runtime.py`, and that scope
  # resolves through mmpose/.mim/configs — so without this, no config loads.
  # Recreate exactly the symlinks develop mode would have made.
  echo "==> Linking mmpose/.mim (configs, tools, model-index.yml, ...)"
  MIM_DIR="${REPO_ROOT}/mmpose/.mim"
  mkdir -p "$MIM_DIR"
  for name in tools configs demo model-index.yml dataset-index.yml; do
    if [[ -e "${REPO_ROOT}/${name}" ]]; then
      rm -rf "${MIM_DIR:?}/${name}"
      ln -s "../../${name}" "${MIM_DIR}/${name}"
    fi
  done
fi

# --------------------------------------------------------------- verify
echo "==> Verifying installation"
# cd so the config path in the model-build check resolves regardless of where
# this script was invoked from.
cd "$REPO_ROOT"
run_in_env python - <<'PY'
import sys

failures = []

def check(label, fn):
    try:
        print(f"  {label:<34} {fn()}")
    except Exception as exc:
        failures.append(label)
        print(f"  {label:<34} FAILED: {type(exc).__name__}: {exc}")

def _torch():
    import torch
    return f"{torch.__version__} (cuda build {torch.version.cuda})"

def _cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    return f"{torch.cuda.get_device_name(0)}"

def _matmul():
    import torch
    a = torch.randn(256, 256, device="cuda")
    return f"ok ({(a @ a).sum().item():.1f})"

def _numpy():
    import numpy
    assert numpy.__version__.startswith("1."), f"numpy {numpy.__version__} is 2.x"
    return numpy.__version__

def _mmcv():
    import mmcv
    return mmcv.__version__

def _mmengine():
    import mmengine
    return mmengine.__version__

def _mmpose():
    import mmpose
    return f"{mmpose.__version__} ({mmpose.__file__})"

def _cocometric():
    from mmpose.evaluation.metrics import CocoMetric  # noqa: F401
    return "importable (xtcocotools ok)"

def _tensorboard():
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    return "importable"

# The four custom modules this fork adds.
def _custom():
    from mmpose.datasets.transforms.rgbd import (  # noqa: F401
        LoadRGBDImage, PhotometricDistortionRGBOnly)
    from mmpose.models.data_preprocessors.nchannel import (  # noqa: F401
        NChannelPoseDataPreprocessor)
    from mmpose.models.losses.geometric_loss import (  # noqa: F401
        GeometricKeypointMSELoss)
    from mmpose.engine.vis_backends.safe_mlflow import (  # noqa: F401
        SafeMLflowVisBackend)
    return "rgbd + nchannel + geometric_loss + safe_mlflow"

def _sklearn():
    import sklearn
    return sklearn.__version__

def _onnx():
    import onnx, onnxruntime
    return f"onnx {onnx.__version__} / ort {onnxruntime.__version__}"

# Build the actual HRNet model from the RGBD+geom config — the strongest
# end-to-end signal that training will start. `custom_imports` in the config
# is honoured by Runner, not by Config.fromfile, so import the modules here.
def _build_model():
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    import mmpose.datasets.transforms.rgbd            # noqa: F401
    import mmpose.models.data_preprocessors.nchannel  # noqa: F401
    import mmpose.models.losses.geometric_loss        # noqa: F401
    from mmpose.registry import MODELS

    init_default_scope("mmpose")
    cfg = Config.fromfile("configs/hrnet-w32_8-kp_udp_rgbd_geom.py")
    # Skip the pretrained-weight fetch; we only care that it assembles.
    cfg.model.backbone.init_cfg = None
    model = MODELS.build(cfg.model)
    n = sum(p.numel() for p in model.parameters())
    return f"HRNet-W32 RGBD+geom built ({n/1e6:.1f}M params)"

check("python",                 lambda: sys.version.split()[0])
check("numpy (<2 required)",    _numpy)
check("torch",                  _torch)
check("cuda device",            _cuda)
check("cuda matmul",            _matmul)
check("mmengine",               _mmengine)
check("mmcv",                   _mmcv)
check("mmpose",                 _mmpose)
check("CocoMetric",             _cocometric)
check("tensorboard",            _tensorboard)
check("custom modules",         _custom)
check("scikit-learn",           _sklearn)
check("onnx / onnxruntime",     _onnx)
check("build HRNet from config", _build_model)

print()
if failures:
    print(f"FAILED ({len(failures)}): {', '.join(failures)}")
    sys.exit(1)
print("All checks passed.")
PY

cat <<EOF

==> Done. To activate, initialise your shell once:

      micromamba shell init --shell bash --root-prefix="\${MAMBA_ROOT_PREFIX:-\$HOME/micromamba}"
      exec bash

    NOTE: pass --root-prefix as shown. micromamba's own hint suggests
    ~/.local/share/mamba, which is *not* where this env lives, and
    'activate' would then fail to find it.

    Then, in any new shell:
      micromamba activate ${ENV_NAME}

    Or skip activation entirely for one-off commands:
      micromamba run -n ${ENV_NAME} python tools/train.py <config>

    Train:
      python tools/train.py configs/hrnet-w32_8-kp_udp_rgbd_geom.py

    Before the RGBD configs will run you still need:
      - data/RSC_Keypoints_RGBD/{images,depth,annotations}
      - work_dirs/pretrained/hrnet_w32_rgbd.pth
          python scripts/convert_hrnet_to_rgbd.py
EOF
