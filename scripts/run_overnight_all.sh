#!/bin/bash
# Full overnight chain, meant to run inside a persistent tmux/byobu session
# (survives Claude Code / SSH disconnect; only needs WSL to stay up).
#   1) CIFAR10 isolated 2x2          (run_iso_2x2.sh,        ~5h)
#   2) CIFAR10 isolated 2x2 DENSE    (run_iso_dense.sh,      ~7.5h)
#   3) CIFAR100 teacher + dispersion + isolated 2x2 (run_c100_overnight.sh, ~6h)
set -u
cd /mnt/d/research/FAT
echo "================ OVERNIGHT CHAIN START $(date) ================"
bash run_iso_2x2.sh
bash run_iso_dense.sh
bash run_c100_overnight.sh
echo "================ OVERNIGHT CHAIN ALL DONE $(date) ================"
