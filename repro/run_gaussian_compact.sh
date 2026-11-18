#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-data/imagenet_val}" != data/imagenet_val ]]; then
    echo 'For a custom ImageNet path, use gaussian.py train --profile compact --epochs 20 --data PATH.' >&2
    exit 2
fi
exec bash run_epoch128.sh imagenet "${2:-256}" "${3:-20}"
