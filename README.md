# FAT — Feature-Direction Adversarial Distillation

Adversarial training where a student is distilled from a naturally-trained teacher in
**feature-direction space** instead of logit space: the teacher only tells the student which
way its (L2-normalized) feature should point, magnitude/confidence is left as the student's own
property.

## Environment

Conda env `advTrain`. Key deps: `torch`, `torchattacks`, `robustbench`/`autoattack` (vendored
under `Externals/`), `torchvision`, `sklearn`, `pandas`, `seaborn`.

## Running an experiment

```bash
python -u main.py --config_name <name>.yaml --dataset CIFAR100 --seed 0 \
    [--eta <k>] [--lamda <l>] [--kappa <k>] [--tau <t>] [--beta <b>] ...
```

- `--config_name` picks a yaml under `config/<dataset>/`.
- Any of `--alpha/--lamda/--beta/--gamma/--eta/--tau/--kappa/--epochs/--lr/...` overrides the
  matching key in the yaml (sentinel `-1`/`None` = "use yaml value").
- Logs go to `results/<dataset>/<config_name>/output.log`; checkpoints to
  `<dataset>/checkpoint/<config_name>/`. **Two runs sharing the same `--config_name` write to
  the same log/checkpoint path** — don't run CLI-override sweeps of the same yaml concurrently.

## Current best (champion)

Config: [`config/CIFAR100/featdir_span_random_10step_wa.yaml`](config/CIFAR100/featdir_span_random_10step_wa.yaml)

```bash
python -u main.py --config_name featdir_span_random_10step_wa.yaml \
    --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0
```

(`method: feat_direction` in the yaml dispatches to `train_feat_direction` in `methods.py`.)

**Loss** (backbone / head split supervision):

```
L = || Phi_hat_s(x_adv) - Phi_hat_t(x) ||^2                          backbone, direction-only
  + beta * KL( head(scale * Phi_hat_s.detach()) || z_t / tau )       head-only (detached)
  + lamda * KL( head(Phi_hat_s(x_adv)) || head(Phi_hat_s(x)).sg )    consistency (adv <-> clean)
```

`Phi_hat_s`, `Phi_hat_t` are L2-normalized student/teacher features (teacher post-hoc, no
gradient). The backbone never sees the teacher's classifier head — only its normalized feature
direction. The PGD attack (10 steps, step_size 2/255, eps 8/255) maximizes the same direction
loss the backbone trains on (`inner_featdir_only_return` in `utils.py`), projected onto a
`k=350`-dim random subspace of the 512-dim feature space (`featdir_span: random`, `--eta 350`).
The student's own weight-averaged (EMA) copy is evaluated, with decay gliding from
`kappa=0.999` toward 1.0 over training.

**Results** (CIFAR100, ResNet18, seed 0):

| clean | PGD-20 | PGD-10 | CW | AutoAttack |
|-------|--------|--------|----|------------|
| 62.75 | 33.96  | 34.18  | 28.41 | 26.29 |

H(pgd) 44.07, H(cw) 39.11, NRR 37.06 (harmonic mean of clean/robust — always report clean
alongside robust, it's a trade-off, not a single number).

Caveat: AutoAttack is only run on this single winner (`aa: False` in sweep configs by
convention; only re-verify AA after picking a candidate).
