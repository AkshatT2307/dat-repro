#!/usr/bin/env bash
# Fresh Linux workstation setup. Does not install system drivers or expose HF tokens.
set -euo pipefail
cd "$(dirname "$0")"
RUN_AFTER=0
case "${1:-}" in
    '') ;;
    --run) RUN_AFTER=1 ;;
    --help) echo 'Usage: bash setup_steps50k.sh [--run]'; echo 'Default: install environment, fetch data, check all four training plans. --run also launches training.'; exit 0 ;;
    *) echo 'Usage: bash setup_steps50k.sh [--run]' >&2; exit 2 ;;
esac
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.13 || command -v python3)}"
"$PYTHON_BIN" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), "Use Python 3.10–3.13; Python 3.13 matches the original environment"'
VENV_DIR="${VENV_DIR:-$PWD/.venv_steps50k}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then "$PYTHON_BIN" -m venv "$VENV_DIR"; fi
PY="$VENV_DIR/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -r requirements-steps50k.txt
"$PY" -m pip check
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable. Install a compatible NVIDIA driver before training.'
x=torch.randn(32,32,device='cuda'); (x@x).sum().item()
print('GPU:',torch.cuda.get_device_name(), '| PyTorch:',torch.__version__)
PY
mkdir -p logs
"$PY" -m pip freeze > logs/steps50k_environment.lock.txt
# ImageNet is gated: approval must be granted by the account holder on Hugging Face.
if ! "$PY" -c 'from huggingface_hub import get_token; raise SystemExit(0 if get_token() else 1)'; then
    echo 'Accept access at https://huggingface.co/datasets/ILSVRC/imagenet-1k, then authenticate.'
    if [[ -t 0 ]]; then
        "$VENV_DIR/bin/hf" auth login
    else
        echo "Set HF_TOKEN securely or run $VENV_DIR/bin/hf auth login, then rerun setup." >&2
        exit 1
    fi
fi
"$PY" -u download_imagenet.py
"$PY" -u imagenet_split.py
"$PY" -u prepare_data.py --benchmarks
"$PY" -u prepare_seismic.py
for DATASET in imagenet marmousi2; do
    DATA=data/imagenet_val
    if [[ "$DATASET" == marmousi2 ]]; then DATA=data/marmousi2; fi
    for METHOD in idr supervised; do
        "$PY" train_steps50k.py --dataset "$DATASET" --method "$METHOD" --data "$DATA" --dry-run
    done
done
printf '\nSetup finished. Four runs are ready: 128x128, batch 4, exactly 50,000 steps each.\n'
if [[ "$RUN_AFTER" == 1 ]]; then
    TRAIN_PYTHON="$PY" bash run_steps50k.sh
else
    printf 'Start with: TRAIN_PYTHON=%q bash run_steps50k.sh\n' "$PY"
fi
