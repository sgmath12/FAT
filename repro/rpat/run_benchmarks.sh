#!/usr/bin/env bash
# Consistency-AT + RPAT (the paper's Sec. 5.2 setting: RPAT_Benchmarks, no weight averaging), 2026-09-16.
#
#   DATASET=cifar100 bash run_benchmarks.sh          # 110 epochs, ResNet-18, Linf 8/255
#   DATASET=cifar10  bash run_benchmarks.sh
#   EVAL_ONLY=1 LOAD_PATH=<logs dir>/last.model DATASET=cifar100 bash run_benchmarks.sh
#
# Training and the AutoAttack evaluation both write to logs/ next to this script; the run directory
# under RPAT_Benchmarks/logs is named by the dataset, model and mode.
set -euo pipefail
ENV_NAME=${ENV_NAME:-advTrain}
GPU=${GPU:-0}
DATASET=${DATASET:-cifar100}
MODEL=${MODEL:-resnet18}
EPOCHS=${EPOCHS:-110}
EPSILON=${EPSILON:-0.03137254901960784}   # 8/255
ALPHA=${ALPHA:-0.00784313725490196}       # 2/255
SEED=${SEED:-0}
EVAL_ONLY=${EVAL_ONLY:-0}
LOAD_PATH=${LOAD_PATH:-}
DATA_PATH=${DATA_PATH:-/mnt/d/research/FAT/data/$( [[ "$DATASET" == "cifar10" ]] && echo CIFAR10 || echo CIFAR100 )}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/RPAT_Benchmarks"
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 RPAT_DATA_PATH="$DATA_PATH"
# advertorch ships with FAT rather than the environment
export PYTHONPATH="${PYTHONPATH:-}:/mnt/d/research/FAT/Externals"

mkdir -p "$SCRIPT_DIR/logs"
LOG="$SCRIPT_DIR/logs/consistency_ra_${DATASET}_${MODEL}_$(date +%Y%m%d-%H%M%S).log"
echo "[run_benchmarks.sh] dataset=$DATASET data=$DATA_PATH epochs=$EPOCHS log -> $LOG"

if [[ "$EVAL_ONLY" == "1" ]]; then
    [[ -n "$LOAD_PATH" ]] || { echo "EVAL_ONLY needs LOAD_PATH" >&2; exit 1; }
    python eval.py --mode test_auto_attack --model "$MODEL" --distance Linf --epsilon "$EPSILON" \
        --dataset "$DATASET" --load_path "$LOAD_PATH" 2>&1 | tee -a "$LOG"
    exit 0
fi

python train.py --mode adv_train --consistency --RA --model "$MODEL" --distance Linf \
    --epsilon "$EPSILON" --alpha "$ALPHA" --epochs "$EPOCHS" --dataset "$DATASET" --seed "$SEED" \
    < /dev/null 2>&1 | tee -a "$LOG"

# The run directory is timestamped, so pick the newest one for this dataset/model and evaluate it.
RUN_DIR="$(ls -1dt logs/*_${DATASET}_${MODEL}_adv_train_pgd_Linf_*consistency_ra_seed_${SEED} | head -1)"
echo "[run_benchmarks.sh] AutoAttack on $RUN_DIR/last.model" | tee -a "$LOG"
python eval.py --mode test_auto_attack --model "$MODEL" --distance Linf --epsilon "$EPSILON" \
    --dataset "$DATASET" --load_path "$RUN_DIR/last.model" < /dev/null 2>&1 | tee -a "$LOG"
