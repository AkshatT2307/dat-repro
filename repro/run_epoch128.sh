#!/usr/bin/env bash
# Fixed epochs: batch changes optimizer-step count, not the number of samples.
set -euo pipefail
cd "$(dirname "$0")"
DATASET="${1:-imagenet}"
BATCH="${2:-256}"
EPOCHS="${3:-20}"
PY=.venv/bin/python
if [[ ! "$BATCH" =~ ^[1-9][0-9]*$ || ! "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'Usage: bash run_epoch128.sh {imagenet|marmousi2|all} [batch=256] [epochs=20]' >&2
    exit 2
fi
if [[ "$DATASET" == all ]]; then
    bash run_epoch128.sh imagenet "$BATCH" "$EPOCHS"
    bash run_epoch128.sh marmousi2 "$BATCH" "$EPOCHS"
    exit
fi
case "$DATASET" in
    imagenet) DATA=data/imagenet_val ;;
    marmousi2) DATA=data/marmousi2 ;;
    *) echo 'Dataset must be imagenet, marmousi2 or all.' >&2; exit 2 ;;
esac
mkdir -p logs
LOG="logs/epoch128-${DATASET}-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec > >(tee -a "$LOG") 2>&1
printf 'Dataset: %s | resolution: 128 | batch: %s | maximum epochs: %s\nLog: %s\n' "$DATASET" "$BATCH" "$EPOCHS" "$LOG"
if [[ "$DATASET" == marmousi2 && ! -f "$DATA/metadata.json" ]]; then
    "$PY" -u prepare_seismic.py
fi
# Probe the more expensive RGB Fast IDR path; it is conservative for single-channel seismic.
if ! "$PY" -u gaussian.py probe --profile compact --batch "$BATCH" --steps 3; then
    echo "Batch $BATCH failed the GPU preflight. Free GPU memory or rerun with batch 128/64; epochs stay fixed." >&2
    exit 1
fi
for METHOD in idr supervised; do
    OUT="runs/epoch128/$DATASET/$METHOD/b${BATCH}_e${EPOCHS}"
    RESUME=()
    if [[ -f "$OUT/latest.pt" ]]; then RESUME=(--resume); fi
    "$PY" -u gaussian.py train --profile compact --dataset "$DATASET" --method "$METHOD" \
        --data "$DATA" --batch "$BATCH" --epochs "$EPOCHS" --output "$OUT" --validate --patience "${PATIENCE:-5}" --min-delta "${MIN_DELTA:-0.00001}" "${RESUME[@]}"
    if [[ "$DATASET" == imagenet ]]; then
        "$PY" -u evaluate.py --weights "$OUT/best_weights.pth" --output "$OUT/evaluation"
    else
        "$PY" -u evaluate_seismic.py --weights "$OUT/best_weights.pth" --data "$DATA" --split validation --output "$OUT/validation"
        "$PY" -u evaluate_seismic.py --weights "$OUT/best_weights.pth" --data "$DATA" --split test --output "$OUT/evaluation"
    fi
done

"$PY" -u compare_epoch128.py --dataset "$DATASET" --batch "$BATCH" --epochs "$EPOCHS"
