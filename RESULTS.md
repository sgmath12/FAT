# Experiment results

Our own runs, scored with this repo's eval code (`utils.evaluate` / `utils.evaluate_final_aa`),
alongside the two baselines we position against. Literature-wide comparisons live in
`comparison_resnet18_cifar10.md` / `comparison_resnet18_cifar100.md`; this file is **our
measurements**.

- Threat model: ℓ∞, **ε = 8/255 at evaluation** (training ε differs per row, see the config).
- **AA** = AutoAttack, `apgd-ce` + `apgd-t`, all 10 000 test images.
- **NRR** = harmonic mean(clean, AA) — ADR's own calibration metric. Computed here from the
  clean and AA columns.
- All rows are the **`last`** checkpoint, **seed 0**, unless stated. `—` = not measured.
- Every comparison reports clean next to robust: this is a trade-off surface, not one number.

---

## 1. CIFAR-100 / ResNet-18

| run | clean | PGD-20 | CW | AA | NRR |
|---|---:|---:|---:|---:|---:|
| **`featdir_champ200_angeps`** (champion) | **62.17** | 34.77 | **30.92** | 28.59 | **39.17** |
| `featdir_champ200_100ep` (prev. champion) | 60.74 | 34.94 | 30.53 | **28.69** | 38.97 |
| `featdir_champ200_trawsnorm` | 60.26 | 35.33 | 30.49 | 28.63 | 38.82 |
| `featdir_champ200_noawp_angeps` | 62.25 | 32.49 | 30.32 | 28.13 | 38.75 |
| `featdir_champ200_headspan` | 60.30 | 35.76 | 30.31 | 28.46 | 38.67 |
| `featdir_champ200_rawfeat` | 59.79 | 34.94 | 30.13 | 28.57 | 38.66 |
| `nofeat_champ200_norm` (no feature loss) | 58.92 | 35.10 | 30.03 | 28.71 | 38.61 |
| `featdir_champ200_noawp` | 60.73 | 33.32 | 29.60 | 28.08 | 38.40 |
| `featdir_champ200_tau1` (τ 16→1) | 60.26 | 31.31 | 29.93 | 27.78 | 38.03 |
| `featdir_champ200_fullraw` | 57.78 | 35.43 | 29.68 | 28.04 | 37.76 |
| `featdir_awp_100ep_eps10` | 60.04 | 34.36 | 29.31 | 27.36 | 37.59 |
| `nofeat_champ200_raw` (diverged) | 45.84 | 29.65 | 24.43 | 22.66 | 30.33 |
| **ADR + WA + AWP** (baseline) | 57.36 | 34.92 | 30.62 | 28.50 | 38.08 |

ADR's clean/PGD/CW are our **local** re-evaluation of ADR's published checkpoint; its AA is ADR's
**published** value (see `baseline.md`).

**Two rows deserve attention rather than the ranking:**

- `nofeat_champ200_norm` removes the feature loss entirely (plain KD on `z_t/τ`, everything else
  the champion) and still reaches **AA 28.71** — a tie with the champion's 28.69. The directional
  loss's net contribution over plain KD is clean **+1.82** and CW **+0.50**, *not* AA. This is the
  largest open threat to the method's story.
- The champion wins NRR on **clean**, not AA: 28.59 vs 28.69 is a loss of 0.10, inside the day's
  observed AA noise band (28.46–28.71).

### 1b. Screening runs (50ep regime, no AA)

Used to choose directions; **PGD margins here did not transfer to AA**, so treat CW as the only
useful proxy in this table.

| run | clean | PGD-20 | CW |
|---|---:|---:|---:|
| `angeps_p10` (p = 1.0) | **64.59** | 28.77 | **27.05** |
| `angeps_p05` (p = 0.5) | 64.36 | 28.45 | 26.99 |
| `fg_plain_th_sh_kl` (p = 0 base) | 62.61 | 29.16 | 26.63 |
| `n2_traw_sraw` | 62.86 | 32.36 | 27.18 |
| `n2_tnorm_snorm` | 62.53 | 29.27 | 26.61 |
| `proto_g025` (target shrink) | 62.08 | 29.27 | 26.76 |
| `etf_rotate` (ETF rotation) | 60.48 | 26.82 | 26.54 |
| `featdir_wa_sgd_lr001` (SGD switch) | 55.43 | 22.97 | 16.32 |

---

## 2. CIFAR-10 / ResNet-18

| run | clean | PGD-20 | CW | AA | NRR |
|---|---:|---:|---:|---:|---:|
| **`featdir_champ200_angeps`** (champion) | **84.66** | 56.74 | **53.94** | 51.87 | **64.33** |
| `featdir_champ200_100ep` | 82.52 | 57.20 | 53.74 | **51.89** | 63.71 |
| `featdir_scratch_200ep` (no warm start) | 80.96 | 55.74 | 51.54 | 49.85 | 61.71 |
| **ADR + WA + AWP** (baseline) | 83.26 | — | — | 51.18 | 63.39 |
| **CURE** (baseline) | 86.76 | 54.92 | 52.48 | 49.69 | 63.19 |

The champion recipe is **identical to the CIFAR-100 one** — the two configs differ only in
`dataset` and the teacher checkpoint path. No per-dataset tuning was done; CIFAR-10 is a
transfer, not a re-tune.

`featdir_scratch_200ep` costs **−2.02 AA** against the warm-started champion, which is what
justifies initializing from the natural teacher rather than from scratch.

---

## 3. CIFAR-10 / WRN-34-10

| run | clean | PGD-20 | CW | AA | NRR |
|---|---:|---:|---:|---:|---:|
| `wrn_champ_angeps_mixupT` (**AA running**) | **88.31** | 58.67 | **56.72** | *pending* | *pending* |
| `featdir_wrn_mixupT` (earlier, no freeze/WA) | 87.12 | 52.50 | 50.67 | — | — |
| **ADR + WA + AWP** (baseline) | 86.11 | — | — | 55.22 | 67.29 |
| **CURE** (baseline) | 87.05 | — | — | 52.10 | 65.19 |

**The bar is NRR 67.29.** At clean 88.31 that requires **AA ≥ 54.35**. The ResNet-18 CW→AA drop on
CIFAR-10 was 2.07 (53.94 → 51.87); the same drop here would land ≈ 54.6, i.e. right on the line.

Caveat to carry into any table that uses this row: it uses a **mixup teacher**, while every
ResNet-18 row above uses a plain teacher — so it moves architecture *and* teacher at once.
`wrn_champ_angeps_plainT` is queued precisely to separate the two.

Budget note: ADR's WRN rows are **200 epochs SGD**; ours is **100 epochs**. Half the schedule.

---

## 4. Baseline reference (published)

| method | arch | dataset | clean | AA | NRR |
|---|---|---|---:|---:|---:|
| ADR | ResNet-18 | CIFAR-10 | 82.41 | 50.39 | 62.54 |
| ADR + WA | ResNet-18 | CIFAR-10 | 82.59 | 50.86 | 62.95 |
| **ADR + WA + AWP** | ResNet-18 | CIFAR-10 | 83.26 | 51.18 † | 63.39 |
| ADR | WRN-34-10 | CIFAR-10 | 84.67 | 53.24 | 65.37 |
| ADR + WA | WRN-34-10 | CIFAR-10 | 82.93 | 54.13 | 65.50 |
| **ADR + WA + AWP** | WRN-34-10 | CIFAR-10 | 86.11 | 55.22 | 67.29 |
| ADR | ResNet-18 | CIFAR-100 | 56.10 | 26.87 | 36.34 |
| ADR + WA | ResNet-18 | CIFAR-100 | 58.30 | 27.54 | 37.41 |
| **ADR + WA + AWP** | ResNet-18 | CIFAR-100 | 57.36 | 28.50 | 38.08 |
| ADR | WRN-34-10 | CIFAR-100 | 59.76 | 29.36 | 39.38 |
| ADR + WA | WRN-34-10 | CIFAR-100 | 57.42 | 30.46 | 39.80 |
| **ADR + WA + AWP** | WRN-34-10 | CIFAR-100 | 62.21 | 31.63 ‡ | 41.94 |
| CURE | ResNet-18 | CIFAR-10 | 86.76 | 49.69 | 63.19 |
| CURE | WRN-34-10 | CIFAR-10 | 87.05 | 52.10 | 65.19 |

Sources: ADR rows from the ADR repo README (`/mnt/d/research/ADR/README.md`) **except the two
marked cells**; CURE ResNet-18 from the CURE paper Table 1, CURE WRN via RPAT Table 4. NRR is
recomputed here from each row's own clean and AA, so it may differ by ≤0.1 from a value quoted
elsewhere that was rounded differently.

† The ADR **paper** reports 51.18 for this cell, the **repo README** 51.24. We keep the paper value
so this file agrees with `baseline.md` and `comparison_resnet18_cifar10.md`.
‡ Likewise 31.60 (paper Table 3) vs 31.63 (README); the README value is used here.

**CURE has no CIFAR-100 WRN number** — its CIFAR-100 table (Table 2) is ResNet-18 only and reports
PGD/C&W without AA.

Architecture check: ADR's `wideresnet-34-10` (`depth=34, width=10`) and ours
(`CIFAR10/models/WideResNet.py`, `depth=34, widen_factor=10`, channels `[16,160,320,640]`) are the
same standard WRN-34-10.

---

## 5. Standing caveats

1. **Seed 0 only**, every row. The clean gains (+1.4~2.1) are large, but the "AA is a tie" claim
   needs at least one more seed to be safe.
2. The angeps ablation **mixes schedules**: the base row is 50ep while the +WA rows are 100ep.
3. **CW does not predict AA.** The 50ep raw/raw cell and AWP both won on PGD/CW at 50ep and then
   failed to carry into the champion recipe. Only AA arbitrates.
4. AA is the arbiter of the headline claim, and on AA we are **tied, not ahead** — see §1.
