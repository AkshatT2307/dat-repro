#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DATA="${1:-data/imagenet_val}"
BATCH="${2:-auto}"
PY=.venv/bin/python
mkdir -p logs
LOG="logs/pipeline-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec > >(tee -a "$LOG") 2>&1
printf 'Dataset: %s\nBatch: %s\nIterations per model: 50000\nPipeline log: %s\n' "$DATA" "$BATCH" "$LOG"
if [[ "$BATCH" == auto ]]; then
    "$PY" -u gaussian.py tune
elif [[ ! "$BATCH" =~ ^[1-9][0-9]*$ ]]; then
    echo 'Batch must be a positive integer or auto.' >&2
    exit 2
fi
"$PY" -u gaussian.py train --method idr --data "$DATA" --batch "$BATCH" --steps 50000
"$PY" -u gaussian.py train --method supervised --data "$DATA" --batch "$BATCH" --steps 50000
"$PY" -u evaluate.py --weights runs/idr/weights.pth --output runs/idr/evaluation
"$PY" -u evaluate.py --weights runs/supervised/weights.pth --output runs/supervised/evaluation
