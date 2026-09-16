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

2026-09-15: `train_cifar_ra.py` also gains `--awp-gamma / --awp-warmup / --awp-proxy-lr` (AWP proxy form,
copied from FAT `utils.py:AdvWeightPerturb`), exposed in `run.sh` as `AWP_GAMMA` / `AWP_WARMUP`.  The
RPAT/BoAT term is refactored into a function of the network so AWP evaluates the method's full loss
on its proxy; with gamma 0 the step is the upstream one.  RPAT++ + AWP is run with
`AWP_GAMMA=0.005 AWP_WARMUP=20` (10% of 200 epochs), its own WA unchanged.

2026-09-16: `--natural-init <FAT checkpoint>` (`NATURAL_INIT` in `run.sh`) warm-starts from our
naturally trained teacher, stripping the `encoder.` prefix and switching input normalization to the
statistics that checkpoint was trained with (the script hardcodes CIFAR-10's on both datasets; with
the teacher's own, a loaded CIFAR-100 teacher measures 77.25% clean against 36.30%).

## Running this on another machine

`train_cifar_ra.py` and `attacks.py` here are the full modified files, so a fresh clone can be brought
up by copying rather than patching:

    git clone https://github.com/FlaAI/RPAT && cd RPAT && git checkout $(cat UPSTREAM_COMMIT)
    cp <this dir>/train_cifar_ra.py <this dir>/attacks.py RPAT_SOTAs/
    cp <this dir>/resnet.py RPAT_SOTAs/networks/resnet.py
    cp <this dir>/run.sh .

The three cells the paper needs, each 200 epochs on ResNet-18 (about eight hours per cell here):

    # RPAT++ as published (its own WA, no AWP)
    NUM_CLASSES=100 DATA_DIR=<FAT>/data/CIFAR100 FNAME=rpatpp_c100_resnet18 bash run.sh

    # + AWP, the rest of our stack
    NUM_CLASSES=100 DATA_DIR=<FAT>/data/CIFAR100 FNAME=rpatpp_awp_c100_resnet18 \
      AWP_GAMMA=0.005 AWP_WARMUP=20 bash run.sh

    # + AWP + our natural warm start
    NUM_CLASSES=100 DATA_DIR=<FAT>/data/CIFAR100 FNAME=rpatpp_awp_natinit_c100_resnet18 \
      AWP_GAMMA=0.005 AWP_WARMUP=20 \
      NATURAL_INIT=<FAT>/CIFAR100/checkpoint/clean_200ep/clean_last.pkl bash run.sh

CIFAR-10 is the same with `NUM_CLASSES=10`, `DATA_DIR=<FAT>/data/CIFAR10` and the CIFAR-10 teacher.
`BACKGROUND=0` keeps a run in the foreground (the default detaches it with setsid); the final table is
printed as `[last wa]` / `[best wa]` in `exps/$FNAME/output.log`.

## Consistency-AT + RPAT (the paper's Sec. 5.2 setting, no weight averaging)

That result comes from the other half of the repository, `RPAT_Benchmarks` (110 epochs, SGD 0.1 with
decays at 100 and 105, no WA), not from `RPAT_SOTAs`.  `run_benchmarks.sh` runs it and then evaluates
the last checkpoint with AutoAttack:

    DATASET=cifar100 bash run_benchmarks.sh
    DATASET=cifar10  bash run_benchmarks.sh

Two edits make it run outside the authors' machine, both carried in `rpat_local.patch` and copied here
as `benchmarks_datasets.py` and `benchmarks_evals.py`:
- `datasets/datasets.py` reads its dataset root from `RPAT_DATA_PATH` instead of a hardcoded `''`.
- `evals/evals.py` imports adversarial-robustness-toolbox lazily; it was a module-level import needed
  only by the C&W evaluation, so every training run required the package.

`advertorch` is not in the environment either; `run_benchmarks.sh` puts FAT's `Externals/` on
PYTHONPATH, and `tensorboardX` was installed into the `advTrain` env.
