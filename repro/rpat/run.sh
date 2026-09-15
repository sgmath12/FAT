#!/usr/bin/env bash
# RPAT++ (ReBAT + RPAT) — CIFAR-10 / PreActResNet18, eps=8/255 Linf
#
# Trains with train_cifar_ra.py (RPAT_SOTAs, README section 3) and, at the end,
# evaluates [last wa] and [best wa] with clean / PGD10 / PGD20 / PGD50 / CW / AA,
# i.e. the same column set FAT's main.py prints.
#
#   bash run.sh                       # train 200ep + final eval
#   RESUME=40 bash run.sh             # continue from model_39.pth/opt_39.pth (see below)
#   EVAL_ONLY=1 bash run.sh           # skip training, just re-evaluate saved checkpoints
#   STRONGER=1 bash run.sh            # ReBAT[strong]: eps 10/255, 12 steps after stage1
#   EPOCHS=100 FNAME=rpatpp_c10_100ep bash run.sh
#   NUM_CLASSES=100 DATA_DIR=/mnt/d/research/FAT/data/CIFAR100 FNAME=rpatpp_c100 bash run.sh
#
# Training runs detached (setsid) by default so closing the terminal cannot SIGHUP it —
# that is what killed the 2026-07-26 run at epoch 45. BACKGROUND=0 keeps it in the
# foreground; EVAL_ONLY implies BACKGROUND=0.
#
# RESUME=N loads model_$((N-1)).pth / opt_$((N-1)).pth, so N must be a multiple of
# CHKPT_ITERS. The LR schedule is recomputed from the epoch index each iteration, so it
# picks up exactly where it left off. What does NOT survive is best_test_robust_acc: it
# resets to 0 and the first resumed epoch overwrites model_best.pth, so this script backs
# the previous bests up to *_best.pre_resume<N>.pth first.
#
# Logs: exps/$FNAME/output.log (written by the script itself, or eval.log in EVAL_ONLY
#       mode) and logs/${FNAME}_<timestamp>.log (full console incl. tqdm progress).
#       The console log is reused and appended to across runs of the same FNAME;
#       NEW_LOG=1 starts a fresh timestamped one.

set -euo pipefail

ENV_NAME=${ENV_NAME:-advTrain}
GPU=${GPU:-0}

# reuse the CIFAR-10 copy FAT already downloaded (torchvision expects <root>/cifar-10-batches-py)
DATA_DIR=${DATA_DIR:-/mnt/d/research/FAT/data/CIFAR10}
NUM_CLASSES=${NUM_CLASSES:-10}

# ResNet18 = the post-activation net our own experiments use (networks/resnet.py, copied from
# FAT), so the numbers are directly comparable. Use MODEL=PreActResNet18 to match the paper.
FNAME=${FNAME:-rpatpp_c10_resnet18}
SAVE_PATH=${SAVE_PATH:-exps}
MODEL=${MODEL:-ResNet18}
EPOCHS=${EPOCHS:-200}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-128}

# threat model — same as FAT: Linf, eps 8/255, PGD-10 with alpha 2/255 during training
EPSILON=${EPSILON:-8}
PGD_ALPHA=${PGD_ALPHA:-2}
ATTACK_ITERS=${ATTACK_ITERS:-10}
NORM=${NORM:-l_inf}

# ReBAT recipe (these are already the script defaults; spelled out for the record):
#   piecewise LR 0.1 with the small 1.5x decay factor at epoch 100/150,
#   weight averaging (decay 0.999) + BoAT/RPAT regularization from epoch 105.
LR_MAX=${LR_MAX:-0.1}
LR_FACTOR=${LR_FACTOR:-1.5}
STAGE1=${STAGE1:-100}
STAGE2=${STAGE2:-150}
BETA=${BETA:-1.0}
DECAY_RATE=${DECAY_RATE:-0.999}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-105}
CHKPT_ITERS=${CHKPT_ITERS:-10}
# AWP on top of the WA RPAT++ already has (2026-09-15); 0 = upstream recipe.  FAT uses gamma 0.005
# with warmup at 10% of epochs, i.e. AWP_GAMMA=0.005 AWP_WARMUP=20 for 200 epochs.
AWP_GAMMA=${AWP_GAMMA:-0}
AWP_WARMUP=${AWP_WARMUP:-0}

RESUME=${RESUME:-0}
EVAL_ONLY=${EVAL_ONLY:-0}
# eval prints a table we want to watch, so it stays in the foreground
[[ "$EVAL_ONLY" == "1" ]] && BACKGROUND=0
BACKGROUND=${BACKGROUND:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/RPAT_SOTAs"

# conda activate (works under `bash run.sh`, which is not a login shell)
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

EXTRA_ARGS=()
[[ "${STRONGER:-0}" == "1" ]] && EXTRA_ARGS+=(--stronger-attack)
[[ "$EVAL_ONLY" == "1" ]] && EXTRA_ARGS+=(--eval)

if [[ "$RESUME" != "0" ]]; then
    EXTRA_ARGS+=(--resume "$RESUME")
    EXP_DIR="$SCRIPT_DIR/RPAT_SOTAs/$SAVE_PATH/$FNAME"
    for f in "model_$((RESUME - 1)).pth" "opt_$((RESUME - 1)).pth"; do
        [[ -f "$EXP_DIR/$f" ]] || { echo "[run.sh] missing checkpoint $EXP_DIR/$f" >&2; exit 1; }
    done
    # best_test_robust_acc restarts at 0, so the first resumed epoch clobbers these
    for f in model_best wa_model_best; do
        [[ -f "$EXP_DIR/$f.pth" ]] && cp -n "$EXP_DIR/$f.pth" "$EXP_DIR/$f.pre_resume$RESUME.pth"
    done
    echo "[run.sh] resuming at epoch $RESUME (backed up *_best.pth -> *_best.pre_resume$RESUME.pth)"
fi

mkdir -p "$SCRIPT_DIR/logs"
# reuse the newest console log for this FNAME so a resumed run appends to the same file
CONSOLE_LOG=""
if [[ "${NEW_LOG:-0}" != "1" ]]; then
    CONSOLE_LOG="$(ls -1t "$SCRIPT_DIR/logs/${FNAME}_"*.log 2>/dev/null | head -1 || true)"
fi
[[ -n "$CONSOLE_LOG" ]] || CONSOLE_LOG="$SCRIPT_DIR/logs/${FNAME}_$(date +%Y%m%d-%H%M%S).log"
echo "[run.sh] env=$ENV_NAME gpu=$GPU data=$DATA_DIR"
echo "[run.sh] exp=$SAVE_PATH/$FNAME  console log -> $CONSOLE_LOG"
{ echo; echo "=== [run.sh] $(date '+%Y-%m-%d %H:%M:%S') start (resume=$RESUME epochs=$EPOCHS) ==="; } >> "$CONSOLE_LOG"

CMD=(python train_cifar_ra.py \
    --fname "$FNAME" \
    --save-path "$SAVE_PATH" \
    --model "$MODEL" \
    --num-classes "$NUM_CLASSES" \
    --data-dir "$DATA_DIR" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --norm "$NORM" \
    --epsilon "$EPSILON" \
    --pgd-alpha "$PGD_ALPHA" \
    --attack-iters "$ATTACK_ITERS" \
    --lr-max "$LR_MAX" \
    --lr-factor "$LR_FACTOR" \
    --stage1 "$STAGE1" \
    --stage2 "$STAGE2" \
    --beta "$BETA" \
    --decay-rate "$DECAY_RATE" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --chkpt-iters "$CHKPT_ITERS" \
    --awp-gamma "$AWP_GAMMA" \
    --awp-warmup "$AWP_WARMUP" \
    "${EXTRA_ARGS[@]}")

if [[ "$BACKGROUND" == "1" ]]; then
    # setsid + nohup: survives the terminal closing, which is what killed the epoch-45 run
    setsid nohup "${CMD[@]}" >> "$CONSOLE_LOG" 2>&1 < /dev/null &
    echo "[run.sh] detached, pid $!"
    echo "[run.sh] follow:  tail -f $CONSOLE_LOG"
    echo "[run.sh] stop:    pkill -f 'train_cifar_ra.py --fname $FNAME'"
    exit 0
fi

"${CMD[@]}" 2>&1 | tee -a "$CONSOLE_LOG"

echo "[run.sh] done. final table:"
grep -E '^\[.*(last|best) wa\]|Mode' -A0 "$SCRIPT_DIR/RPAT_SOTAs/$SAVE_PATH/$FNAME/"*.log | tail -5 || true
