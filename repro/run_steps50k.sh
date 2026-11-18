#!/usr/bin/env bash
# Four new models, sequentially; never resumes an epoch128/continuation5 checkpoint.
set -euo pipefail
cd "$(dirname "$0")"
PY="${TRAIN_PYTHON:-$PWD/.venv_steps50k/bin/python}"
if [[ ! -x "$PY" ]]; then echo 'Run bash setup_steps50k.sh first (or set TRAIN_PYTHON).' >&2; exit 1; fi
ROOT="runs/steps50k_b4_128"
mkdir -p logs
LOG="logs/steps50k_b4_128-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec > >(tee -a "$LOG") 2>&1
printf 'Four sequential runs | 128x128 | batch 4 | 50000 updates each\nResults: %s\nLog: %s\n' "$ROOT" "$LOG"
for DATASET in imagenet marmousi2; do
    DATA=data/imagenet_val
    if [[ "$DATASET" == marmousi2 ]]; then DATA=data/marmousi2; fi
    for METHOD in idr supervised; do
        OUT="$ROOT/$DATASET/$METHOD"
        RESUME=()
        if [[ -f "$OUT/latest.pt" ]]; then RESUME=(--resume); fi
        "$PY" -u train_steps50k.py --dataset "$DATASET" --method "$METHOD" --data "$DATA" \
            --steps 50000 --output "$OUT" --workers "${WORKERS:-4}" "${RESUME[@]}"
        for SELECTION in final best; do
            WEIGHTS="$OUT/weights.pth"
            if [[ "$SELECTION" == best ]]; then WEIGHTS="$OUT/best_weights.pth"; fi
            if [[ "$DATASET" == imagenet ]]; then
                "$PY" -u evaluate.py --weights "$WEIGHTS" --output "$OUT/evaluation_$SELECTION"
            else
                "$PY" -u evaluate_seismic.py --weights "$WEIGHTS" --data "$DATA" --split test --output "$OUT/evaluation_$SELECTION"
            fi
        done
    done
done
"$PY" -u summarize_steps50k.py
printf '\nAll four runs and evaluations finished. See reports/steps50k_b4_128/RESULTS.md\n'
