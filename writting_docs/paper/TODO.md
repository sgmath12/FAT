# TODO — what is in flight, and what code produces it

Updated 2026-09-02, 00:00. Branch `awp-longschedule-20260730`. Everything below runs from
`/mnt/d/research/FAT` as `python main.py --config_name <cfg>.yaml --dataset <DS> --seed 0`, which
dispatches to `getattr(methods, "train_" + config.method)` (`main.py:153`).

---

## 1. Paper sections

| file | state | left to do |
|---|---|---|
| `0_Abstract.tex` | drafted | numbers track the shipped cell; re-check after the AWP re-run |
| `1_Intro.tex` | drafted | contribution bullets run 5 lines each against the references' 3 |
| `2_Analysis.tex` | drafted | Prop. 1 discussion can lose a paragraph |
| `3_Method.tex` | drafted | box-constraint equation (7) is a candidate for the appendix |
| `4_Experiments.tex` | drafted | blanks fill as runs land |
| `5_Appendix.tex` | drafted | — |
| `6_Related.tex` | **stub** | structure to be set by the authors; the header comment lists the intended coverage |
| Conclusion | **missing** | — |

**Length.** Compiled main body was 12 pages against ICLR's 9 (references started on p13 of
`ICLR2027 (1).pdf`). Since then: Algorithm 1 moved to the appendix, Figure 1 added, seven tables moved
to the appendix, the two per-dataset main tables merged into one. Recompile to re-measure. Remaining
trims identified: the box-constraint equation, the corollary discussion, the contribution bullets,
and turning `tab:linearity` and `tab:teacherladder` into figures so they cost less than tables.

---

## 2. Figures

| | content | source | state |
|---|---|---|---|
| **Fig. 1** | frontier, clean vs AA, our operating curve | `scripts/make_fig1_frontier.py` → `figure/frontier.pdf` | **done**; regenerate when $\varepsilon=6,7$ land |
| Fig. 1a | concept diagram: teacher at $x$, student at $x_{\mathrm{adv}}$, $L/F/O$ on the ball | TikZ | to draw |
| Fig. 2 | two-feature toy: $b^\star$ against $\varepsilon$, the exact-zero threshold | Appendix A.1 numerics; CPU only | to write |
| Fig. 3 | teacher-length ladder as a curve | `tab:teacherladder` data | would replace a table |
| Fig. 4 | local linearity, IGDM Fig. 2 style | `scripts/measure_local_linearity.py` | would replace a table |

---

## 3. Runs in flight

Five queues, strictly sequential; each waits on the previous with `pgrep`. Drivers in `scripts/`,
waiters in the session scratchpad, logs in `logs/*_driver.log`.

### Queue 1 — CIFAR-10 baselines *(running)*
`scripts/run_c10_baselines_20260901.sh`

| config | `methods.py` | knobs | fills |
|---|---|---|---|
| `ladder_p0_100ep` | `train_feat_direction` | `finetune`, no WA/AWP, $p{=}0$ | `tab:natteacher` last blank |

Five earlier cells in this queue are done: `at_teacherinit_matched` (`train_madry_at`),
`ard_nat100ep` (`train_ard`), `rslad_nat100ep` (`train_rslad`), `adaad_nat100ep` (`train_adaad`),
`adaadigdm_nat100ep` (`train_adaad_igdm`, $\alpha=20$).

### Queue 2 — AWP-fix re-runs, 9 cells + 1
`scripts/run_awpfix_rerun_20260901.sh`

All `train_feat_direction`. Re-measured because `_awp_loss_fn` included a head term the model never
trains, supplying 40.8% of the AWP backbone gradient; fixed in `021f0ef`
(`methods.py`, the `if freeze_head: return l_` branch and the hoisted `freeze_head` block).

| config | dataset | distinguishing knob | pre-fix value |
|---|---|---|---|
| `adaadigdm_nat100ep_a1` | C10 | `igdm_alpha 1.0` | — (new cell) |
| `l2_bestrecipe_freezehead` | C100 | `freeze_head`, $\varepsilon_{tr}$ 8.8 | 62.65 / 28.77 |
| `champ_eps88` | C10 | same | 85.58 / 51.79 |
| `l2_bestrecipe_angeps` | C100 | no `freeze_head` | 62.35 / 28.68 |
| `ladder_angeps_waawp_100ep` | C100 | ladder top row | 62.35 / 28.68 |
| `ladder_angeps_waawp_50ep` | C100 | ladder top row, 50 ep | 59.90 / 28.05 |
| `champ_eps8`, `champ_eps10` | C100 | $\varepsilon_{tr}$ 8, 10 | 63.89 / 27.78, 59.99 / 28.77 |
| `champ_eps8`, `champ_eps10` | C10 | $\varepsilon_{tr}$ 8, 10 | 87.22 / 51.15, 83.29 / 51.79 |

Cells without `awp_gamma` never enter that closure and are final; the ladder's other six rows are not
re-run for that reason.

### Queue 3 — ablations, 7 cells
`scripts/run_ablations_20260901.sh`, all `train_feat_direction`, all base regime (50 ep, no WA, no
AWP, $\varepsilon = 8/255$) so nothing absorbs the effect. Rationale per claim in
`notes/ablation_plan.md`.

| config | knob | what it tests | read against |
|---|---|---|---|
| `abl_teacher_at_adv` | `featdir_teacher_at_adv` → `utils.inner_featdir_teacher_at_adv` | **the central claim**: teacher read at $x_{\mathrm{adv}}$ instead of $x$ | `ladder_p0_50ep` 61.33 / 26.19 |
| `abl_ce_lam01/03/10` | `featdir_head_ce`, `featdir_alpha 1.0`, `beta` = $\lambda$ | anchor **replaces** CE rather than supplementing it | pure CE 57.73 / 26.46 vs anchor 61.33 / 26.19 |
| `abl_angeps_p05`, `p20` | `featdir_angeps_p` 0.5, 2.0 | is $p{=}1$ a plateau or a tuned point | $p{=}0$ and $p{=}1$ already measured |
| `abl_scratch_init` | `finetune: False`, 100 ep | initialization separated from the anchor | `ladder_p0_100ep` 61.21 / 25.24 |

⚠ `abl_teacher_at_adv` carries a **prediction recorded before the run**, in its config header: clean
and AA both fall, and $O$ on the resulting student rises toward the teacher's 6.68. If it ties,
§2.3 is a description rather than a mechanism and must be rewritten.

### Queue 4 — standard baselines, 6 cells
`scripts/run_std_baselines_20260902.sh`. Implemented here rather than quoted, so the main table is
entirely our own measurements.

| config | `methods.py` | published form |
|---|---|---|
| `pgdat_100ep` | `train_madry_at` | label CE on $x_{\mathrm{adv}}$, random init |
| `trades_100ep` | `train_trades` (+ `_trades_inner_attack`) | $\mathrm{CE}(f(x),y) + \beta\,\mathrm{KL}$, $\beta = 6$ |
| `mart_100ep` | `train_mart` | boosted BCE + $(1-p_y(x))$-weighted KL, $\lambda = 5$ |

Both datasets. Protocol shared with the distillation baselines: SGD 0.1, step decay at epochs 70 and
90, 100 epochs, $\varepsilon = 8/255$, 10-step attack, random init, no WA, no AWP.

### Queue 5 — extend the operating curve, 4 cells
`scripts/run_lowerps_20260902.sh`. `champ_eps6`, `champ_eps7` on both datasets, `train_feat_direction`
with `train_eps` 6/255 and 7/255 and nothing else changed; evaluation stays at 8/255.

Purpose is Figure 1, not a table. At $\varepsilon_{tr} = 8$ the curve already dominates Generalist++
on CIFAR-100 and CURE and ARREST on CIFAR-10; the open case is Generalist++'s CIFAR-10 clean accuracy
of 89.09, and the measured trend (clean $-1.9$ per $+1/255$, AA flat) puts $6/255$ near clean 91.

---

## 4. Table grid

`notes/fill_grid.md` has the full picture. Summary: 33 empty cells, ≈329 GPU-hours to fill, per-cell
cost ×4.0 on Tiny-ImageNet and ×4.5 on WideResNet. A four-row subset — PGD-AT, AdaAD,
PGD-AT-at-teacher-init, ours — costs ≈104 hours and preserves every claim the tables are used for.

**Blocking:** the Tiny-ImageNet result (55.16 / 20.54) is documented but its log is on the other
server, so `results/TinyImageNet/` here has no AutoAttack line. Copy it across.

**Running elsewhere:** WideResNet-34-10, both CIFAR datasets, `wrn_champ_freezehead`. That branch has
the AWP fix (`021f0ef`) and the CIFAR-100 WideResNet fix (`c5d4458` — before it, `arch: WideResNet` on
CIFAR-100 silently built an 11.2M ResNet-18 rather than the 48.3M WideResNet).

---

## 5. Open decisions

| | |
|---|---|
| **Related Work** | structure to be set by the authors; more references to be added |
| **LBGAT** | no paper; its two table cells are quoted from ARREST and ADR and are WideResNet |
| **Conclusion** | not written |
| **ADR** | repo at `/mnt/d/research/ADR`, never run; it is one of the two remaining blanks in the main table |
| **CURE** | seven runs, none reproduce; closed as quoted-with-explanation in `app:cure` |
| **Remaining page overshoot** | after recompiling, decide between the trims listed in §1 |
