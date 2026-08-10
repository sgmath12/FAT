#!/usr/bin/env bash
# Guidance grid (2026-08-01). Normalization is applied ONLY where the teacher builds its guidance;
# the student is vanilla end to end. Backbone follows the feature guidance, head follows the logits.
#
#   teacher A = clean_200ep      (trained with NO normalization, 77.59)
#   teacher B = clean_cos200ep   (trained WITH a cosine classifier: z and w both normalized)
#   guidance 1 = feature only    -> W_t  Phi_hat_t
#   guidance 2 = feature AND w   -> W_hat_t Phi_hat_t   (cosine)
#
# Order: A1, A2, train teacher B, B1, B2.  ~1h per featdir cell, ~30min for the teacher.
# Waits for any in-flight run first.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

run () {  # run <config> <logname>
  echo "=== $(date +%H:%M) start $2 ==="
  $PY main.py --config_name "$1" --dataset CIFAR100 --seed 0 > "logs/$2.log" 2>&1
  echo "=== $(date +%H:%M) done $2 (exit $?) ==="
}

while pgrep -f "main.py --config_name" > /dev/null; do sleep 60; done

run guide_A1.yaml       guide_A1
run guide_A2.yaml       guide_A2
run clean_cos200ep.yaml clean_cos200ep     # teacher B
run guide_B1.yaml       guide_B1
run guide_B2.yaml       guide_B2
echo "=== guidance grid complete $(date) ==="
