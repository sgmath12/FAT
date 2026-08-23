#!/usr/bin/env bash
# ROBUST LINEAR PROBE (2026-08-20).  Refit ONLY the head, identically for every cell, so the
# direction-vs-L2 comparison reduces to the backbone representation.
#
# The backbone was already clean: `featdir_alpha` defaults to 0 (methods.py:2106), so the head KD
# term is fully detached and never touches the backbone.  The HEAD was not clean: it is fitted
# against z_t/16 while reading a feature whose norm is ~1 for the directional design and ~12 for
# raw L2, so the two designs' heads were fitted under materially different conditions and a
# difference in final accuracy could come from either the representation or the head fit.
#
# Here every head is re-initialized (same seed) and refit with adversarial CE for the same 20
# epochs, same optimizer and schedule, backbone frozen in eval mode so BN stats are frozen too.
# The label target removes the external scale the head could mismatch.  Each design keeps its OWN
# inference path -- direction reads Phi_hat, full raw reads Phi -- so this is not the partial-raw
# hybrid; only the head-fitting procedure is equalized.
#
# Cells, and what they scored with their originally trained heads
# (C100 / ResNet18 / 100ep / seed 0, clean / AA):
#   bare      direction `b2x2_snorm_tnorm` 61.52 / 22.90   L2 `wadec_raw_nowa` 62.40 / 24.34
#   champion  direction `wadec_dir_full`   60.62 / 28.55   L2 `wadec_raw_full` 57.96 / 28.13
#
# The bare pair is the one that matters: raw L2 wins there outright, with no stack available to
# explain it away.  If direction still loses after an equalized head fit, the deficit is in the
# representation; if the gap closes, it was in the head.
#
# NOTE `b2x2_snorm_tnorm` predates the 2026-08-17 `main.py` checkpoint fix, so its `_last.pkl` is
# the epoch-95 model rather than epoch-99.  Acceptable here (we are comparing representations, and
# the probe retrains the head anyway) but it is not an exactly matched pair with the other three.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

run () {
  echo "=== $(date '+%m-%d %H:%M') start probe $1 ==="
  $PY -u scripts/head_probe.py --cell "$1" \
      --ckpt "CIFAR100/checkpoint/$1/feat_direction_last.pkl" --epochs 20 \
      > "logs/probe_$1.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done probe $1 (exit $?) ==="
}

run b2x2_snorm_tnorm   # bare direction
run wadec_raw_nowa     # bare L2
run wadec_dir_full     # champion direction
run wadec_raw_full     # champion L2

echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
