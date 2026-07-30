# AWP in this repo: implementation, knobs, and what it actually did

Adversarial Weight Perturbation (Wu, Xia, Wang, NeurIPS 2020) — a flat-minima regularizer that
finds a norm-bounded **weight** perturbation maximizing the (already input-adversarial) training
loss, then takes the descent step at those perturbed weights. It is the standard
robust-generalization / anti-robust-overfitting lever, and it is where our #1 baseline gets its
edge: ADR's own ablation goes AA **26.87 → 28.50** when it stacks WA+AWP.

## Two implementations (`utils.py`)

Both take the *caller's own* training loss rather than the paper's plain CE, matching this
project's matched-adversary discipline (for `feat_direction` that is `dir_loss + head KL`).

### 1. `AdvWeightPerturb` — `awp_style: proxy` (the original paper's mechanics)
Keeps a separate **proxy network**. Per step:
```
diff = awp.calc_awp(loss_fn)   # proxy <- model weights; one SGD ASCENT step on -loss_fn(proxy)
awp.perturb(diff)              # model += gamma * diff
<backward + optimizer.step()>  # the real update, computed AT the perturbed weights
awp.restore(diff)              # model -= gamma * diff
```
`_awp_diff_in_weights` builds the perturbation direction from the proxy-vs-model weight delta,
**norm-scaled per layer** (`||w|| / ||d|| * d`) and restricted to >=2D `weight` tensors — i.e.
conv/linear matrices only, never BN or biases. So `gamma` is a *relative* perturbation size.

### 2. `AdvWeightPerturbSAM` — `awp_style: sam` (the ADR repo's `src/util/awp.py`)
Structurally a **SAM** (Foret et al. 2020) wrapper, not the paper's proxy construction: no second
network, it perturbs the model's own parameters using the gradient already backprop'd at the
current weights, scaled per-parameter. One hyperparameter (`rho`) instead of two
(`proxy_lr` + `gamma`). Needs `_awp_bn_disable`/`_awp_bn_enable` so only the first of its two
forwards updates BN running stats (mirrors ADR's `util/bypass_bn.py`).

## Config knobs
```yaml
awp_style  : proxy    # 'proxy' (Wu et al.) | 'sam' (ADR-repo port)
awp_gamma  : 0.005    # relative perturbation size; = rho for the sam style. 0/absent = AWP OFF
awp_warmup : 10       # epoch index at which AWP switches on
```
Read in `methods.py` (`train_feat_direction`, `train_temperature`); also overridable as
`--awp_gamma` / `--awp_warmup`. AWP is **off by default** (`awp_gamma` 0), so every pre-existing
config behaves exactly as before.

Cost check: proxy AWP adds one forward+backward on top of the ~11 the 10-step attack already
does, so expect **~+10% wall-clock**, not 2x. Our 100ep cells: 1h55 without → 2h06 with.

### Gotcha that already bit us twice
1. `from utils import *` in `methods.py` **silently drops underscore-prefixed names** (Python
   wildcard-import convention), so `_awp_bn_disable`/`_awp_bn_enable` were never imported and the
   `sam` style crashed with `NameError` the instant AWP activated. Fixed with an explicit
   `from utils import _awp_bn_disable, _awp_bn_enable`.
2. The smoke gate used `epochs: 1` and so **never reached `awp_warmup: 5`** — it passed while the
   feature was broken. Any smoke test for an epoch-gated feature must run *past* the gate epoch.

## What AWP actually did (CIFAR100, ResNet18, feat_direction k350+WA+lamda4, eval eps 8/255)

### At 50 epochs — a NULL, because there is nothing to fix
| cell | AA |
|---|---:|
| k350+WA (no AWP) | **26.29** |
| k350+WA+AWP proxy (gamma 0.005 / 0.01 / 0.02) | 25.88 / 25.54 / 24.61 |
| baseline+WA (no AWP) | 26.42 |
| baseline+WA+AWP (gamma 0.005 / 0.01 / 0.02) | 26.40 / 26.45 / 25.49 |

AWP is flat-to-negative **in both pipelines**, so this was never a `feat_direction`-specific
problem. Explanation: AWP suppresses robust overfitting, and the 50ep champion barely overfits
(PGD-20 peaks 34.32 at epoch 45, ends 33.96 — only −0.36). No disease, no cure.

### At 100 epochs — AWP works, on every metric including AA
| cell | clean | FGSM | PGD-20 | PGD-50 | CW | AA |
|---|---:|---:|---:|---:|---:|---:|
| champion, 50ep | 62.75 | 36.48 | 33.96 | 33.93 | **28.41** | **26.29** |
| 100ep control (no AWP) | 62.04 | 35.82 | 32.38 | 32.44 | 27.23 | 25.16 |
| **100ep + AWP** (proxy, g0.005, warmup10) | **63.07** | **36.79** | **34.12** | **34.09** | 28.20 | 25.98 |

Robust overfitting is real at 100ep — vs the 50ep champion the control loses PGD 1.58, CW 1.18,
and **AA 1.13**. AWP recovers all of the PGD/CW loss and more (+1.74 PGD, +0.97 CW, +1.03 clean
over the control, beating even the 50ep champion on PGD and clean), and recovers **+0.82 of the
1.13 AA** (~73%).

**Standing conclusion: AWP is correctly implemented and does its stated job at long schedules —
including on AA, where the 50ep null does not hold.** What it has NOT done is make a long schedule
pay: 100ep+AWP is still 0.31 AA short of the 50ep champion, so the extra 2x compute buys nothing
on the headline metric. The AA gap to ADR-full (28.50 / NRR 38.08 vs our 26.29 / 37.06) stays open.

Note on a coincidence worth not misreading: the k=512 mechanism-ablation checkpoint also scored
exactly 25.98. The control's 25.16 rules out an eval-path bug — the collision was chance.

Reproduce:
```bash
python main.py --config_name featdir_k350wa_100ep_awp.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0
python main.py --config_name featdir_k350wa_100ep.yaml     --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0   # control
python scripts/eval_aa_generic.py "label|CIFAR100/checkpoint/featdir_k350wa_100ep_awp/feat_direction_last.pkl"
```
See `baseline.md` for the ADR comparison and `README.md` for the champion recipe.
