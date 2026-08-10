#!/usr/bin/env bash
# featdir champion on the RPAT/ReBAT training protocol (2026-07-26).
#
# What this is FOR: RPAT++ trains 200 epochs from scratch with SGD 0.1 + piecewise decay
# (/1.5 at 100, /1.5^2 at 150) + weight averaging. Our champion trains 50 epochs with AdamW
# 0.021 + OneCycle *initialized from the natural teacher*. This script runs the champion loss
# under THEIR protocol, so the comparison stops being a schedule/initialization comparison:
#   - finetune: False  -> student starts from random init (the "fine-tuning advantage" objection)
#   - SGD 0.1, momentum 0.9, wd 5e-4, lr_schedule piecewise, 200 epochs (their recipe)
#   - WA stays on (kappa 0.999, annealed) -- ours, and they use it too
#   - everything else = the champion verbatim (k, lamda 4.0, tau 16, beta 1.0, 10-step, eps 8/255)
# The teacher is still loaded (load: True); only the STUDENT init changed.
#
#   bash run_featdir_200ep.sh                          # CIFAR10, k=200, seed 0, detached
#   DATASET=CIFAR100 bash run_featdir_200ep.sh         # CIFAR100, k=350
#   ETA=350 SEED=1 bash run_featdir_200ep.sh           # override k / seed
#   BACKGROUND=0 bash run_featdir_200ep.sh             # keep it in the foreground
#
# k defaults per dataset from the measured curves: CIFAR10 200 (07-23 sweep, 82.68/56.45 pgd20
# beats eta350 55.90 and eta512 55.72), CIFAR100 350 (the pinned champion).
#
# Logs: results/$DATASET/ResNet18/$CONFIG_NAME/<timestamp>.log (written by main.py) and
#       logs/${TAG}_<timestamp>.log (full console). Checkpoints: $DATASET/checkpoint/$CONFIG_NAME/.
# NOTE main.py keys log/checkpoint dirs by --config_name, so two concurrent runs of the SAME
# config (e.g. two seeds) would collide -- CONFIG_NAME is settable for that reason.

set -euo pipefail

ENV_NAME=${ENV_NAME:-advTrain}
GPU=${GPU:-0}
DATASET=${DATASET:-CIFAR10}
SEED=${SEED:-0}
LAMDA=${LAMDA:-4.0}
EPOCHS=${EPOCHS:-200}
CONFIG_NAME=${CONFIG_NAME:-featdir_scratch_200ep.yaml}

if [[ -z "${ETA:-}" ]]; then
    if [[ "$DATASET" == "CIFAR100" ]]; then ETA=350; else ETA=200; fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/config/$DATASET/$CONFIG_NAME"
[[ -f "$CONFIG_PATH" ]] || { echo "[run] missing config $CONFIG_PATH" >&2; exit 1; }

# the natural teacher must exist -- get_model() calls exit() with a bare message otherwise
TEACHER=$(grep -E '^checkpoint\s*:' "$CONFIG_PATH" | head -1 | sed 's/.*:[[:space:]]*//')
[[ -f "$SCRIPT_DIR/$TEACHER" ]] || { echo "[run] missing teacher checkpoint $TEACHER" >&2; exit 1; }

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

TAG=${TAG:-featdir200_${DATASET,,}_k${ETA}_s${SEED}}
mkdir -p "$SCRIPT_DIR/logs"
CONSOLE_LOG="$SCRIPT_DIR/logs/${TAG}_$(date +%Y%m%d-%H%M%S).log"

CMD=(python -u "$SCRIPT_DIR/main.py"
     --config_name "$CONFIG_NAME"
     --dataset "$DATASET"
     --seed "$SEED"
     --eta "$ETA"
     --lamda "$LAMDA"
     --epochs "$EPOCHS")

echo "[run] env=$ENV_NAME gpu=$GPU dataset=$DATASET k=$ETA lamda=$LAMDA epochs=$EPOCHS seed=$SEED"
echo "[run] teacher=$TEACHER"
echo "[run] scratch init (finetune: False), SGD 0.1 piecewise (/1.5 @100, /1.5^2 @150), WA on"
echo "[run] console log -> $CONSOLE_LOG"
echo "[run] result log  -> results/$DATASET/ResNet18/${CONFIG_NAME%.yaml}/"

cd "$SCRIPT_DIR"
if [[ "${BACKGROUND:-1}" == "1" ]]; then
    # setsid + nohup so closing the terminal cannot SIGHUP a 200-epoch run
    setsid nohup "${CMD[@]}" >> "$CONSOLE_LOG" 2>&1 < /dev/null &
    echo "[run] detached, pid $!"
    echo "[run] follow:  tail -f $CONSOLE_LOG"
    echo "[run] stop:    pkill -f 'main.py --config_name $CONFIG_NAME'"
else
    "${CMD[@]}" 2>&1 | tee -a "$CONSOLE_LOG"
fi
