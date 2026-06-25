#!/bin/bash
# Run CURE experiments sequentially: GitHub config (A) first, then paper config (B).
# Both start from our [0,1] natural model (clean_no_convert) and use FAT eval.
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

echo "############ Experiment A (GitHub: beta5, 120ep) START $(date) ############"
rm -f results/CIFAR10/cure_no_convert/output.log
$PY -u main.py --config_name cure_no_convert.yaml
echo "############ Experiment A DONE $(date) ############"

echo "############ Experiment B (paper: beta1, 200ep) START $(date) ############"
rm -f results/CIFAR10/cure_no_convert_paper/output.log
$PY -u main.py --config_name cure_no_convert_paper.yaml
echo "############ Experiment B DONE $(date) ############"

echo "############ ALL DONE $(date) ############"
