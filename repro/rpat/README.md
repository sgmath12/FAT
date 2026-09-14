# RPAT++ reproduction (not ported into methods.py)

RPAT++ (ReBAT + RPAT) was run from a clone of the official repository, https://github.com/FlaAI/RPAT,
at the commit in `UPSTREAM_COMMIT`, cloned to `/mnt/d/research/RPAT`.  This directory keeps our local
changes so the runs can be reproduced from a fresh clone:

    git clone https://github.com/FlaAI/RPAT && cd RPAT && git checkout $(cat UPSTREAM_COMMIT)
    git apply rpat_local.patch
    cp resnet.py RPAT_SOTAs/networks/resnet.py
    cp run.sh .
    bash run.sh                                                      # CIFAR-10
    NUM_CLASSES=100 DATA_DIR=<FAT>/data/CIFAR100 FNAME=rpatpp_c100 bash run.sh   # CIFAR-100

What `rpat_local.patch` changes, and nothing else:
- `train_cifar_ra.py`: a `ResNet18` choice (the post-activation ResNet-18 of our own experiments,
  `resnet.py`), and the final evaluation prints clean / PGD10 / PGD20 / PGD50 / CW / AA like main.py.
- `attacks.py`: PGD10, PGD50 and CW added to `run_all`; upstream's AutoAttack call used `norm='L2'`,
  changed to `norm='Linf'` to match the 8/255 L-inf threat model everything else is evaluated under.
