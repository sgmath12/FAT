# TODO — what is in flight, and what code produces it

Updated 2026-09-02, 12:00. Branch `awp-longschedule-20260730`. Everything below runs from
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

**This machine runs ablations.** Tiny-ImageNet and WideResNet are being filled elsewhere (§4), so
the queue here is ordered around what only this machine is doing. One driver,
`scripts/master_queue_20260902.sh`, runs the four remaining queues strictly sequentially — a single
process rather than independent `pgrep` waiters, which put two trainings on one GPU three times.
Logs in `logs/`.

    ablations (9) -> standard baselines (6) -> lower epsilon (4) -> AWP re-runs (7)

The AWP re-runs moved from first to last on the evidence in §4.4: the fix moves NRR by 0.01 on
CIFAR-100 and 0.22 on CIFAR-10, so the seven remaining cells are decimal-place updates costing about
16 hours, and nothing in the paper waits on them.

### Queue 1 — CIFAR-10 baselines *(done)*
`scripts/run_c10_baselines_20260901.sh`

| config | `methods.py` | knobs | result |
|---|---|---|---|
| `ladder_p0_100ep` | `train_feat_direction` | `finetune`, no WA/AWP, $p{=}0$ | **61.21 / 25.24** (CIFAR-100) |

That last cell is the anchor with no stack and no sensitivity-matched $\varepsilon$, and it is the
comparison point the running ablation `abl_ce_nostack` is measured against — see Queue 3.

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

### Queue 3 — ablations, 9 cells *(running)*
`scripts/run_ablations_20260901.sh`, all `train_feat_direction`, all base regime (50 ep, no WA, no
AWP, $\varepsilon = 8/255$) so nothing absorbs the effect. Rationale per claim in
`notes/ablation_plan.md`.

| config | knob | what it tests | read against |
|---|---|---|---|
| `abl_teacher_at_adv` | `featdir_teacher_at_adv` → `utils.inner_featdir_teacher_at_adv` | **the central claim**: teacher read at $x_{\mathrm{adv}}$ instead of $x$ | `ladder_p0_50ep` 61.33 / 26.19 |
| `abl_ce_lam01/03/10` | `featdir_head_ce`, `featdir_alpha 1.0`, `beta` = $\lambda$ | anchor **replaces** CE rather than supplementing it | pure CE 57.73 / 26.46 vs anchor 61.33 / 26.19 |
| `abl_angeps_p05`, `p20` | `featdir_angeps_p` 0.5, 2.0 | is $p{=}1$ a plateau or a tuned point | $p{=}0$ and $p{=}1$ already measured |
| `abl_scratch_init` | `finetune: False`, 100 ep | initialization separated from the anchor | `ladder_p0_100ep` 61.21 / 25.24 |
| `abl_ce_nostack` *(running)* | `method: madry_at` at the anchor's base regime | **does the anchor buy robustness by itself** | `ladder_p0_100ep` 61.21 / **25.24** |
| `abl_ce_attack` | anchor loss, CE attack | if it does, is that the loss or the attack | same |

`abl_ce_nostack` answers the question the paper cannot currently answer. The reported $+2.31$ AA over
cross-entropy is measured across a difference in *stack* as well as loss — the anchor row has neither
weight averaging nor AWP and the cross-entropy row has both. Its config header records the read
before the number lands: **AA $\geq 25.24$** means the anchor buys no robustness on its own and the
gain at full recipe is the stack interacting with it, which is what §2.3 should then say; **AA
$< 25.24$** means the anchor buys robustness by itself and the margin is the number to report.

**Two have landed, and both are decisive — `notes/ablation_results.md` has the full write-up.**

| cell | result | read |
|---|---|---|
| `abl_teacher_at_adv` | **76.11 / AA 0.00** against 61.33 / 26.19 | robustness is not degraded, it is *gone*; reading $\Phi_t$ at $x_{\mathrm{adv}}$ makes $\Phi_s = \Phi_t$ optimal and that solution is a natural model |
| `abl_ce_nostack` | **54.60 / AA 19.45** against the anchor's 61.21 / 25.24 | the anchor buys **+5.79 AA and +6.61 clean by itself**, no $\varepsilon$-matching, no WA, no AWP |

⚠ `abl_teacher_at_adv`'s recorded prediction was **half wrong**: it said clean would fall along with
AA, and clean *rose* 14.78 to just under the teacher. The mechanism is degeneracy, not instability —
the target becomes trivially trackable rather than noisy. §3.3 must say degeneracy.

**This changes the framing.** "The anchor keeps accuracy and does not cost robustness; WA and AWP
supply the robustness" understates the measurement: at matched stack the anchor supplies +5.79 AA,
against the +2.40 separating the shipped recipe from cross-entropy *with* the stack. Nor is it CE
overfitting — CE's PGD peaks at 23.31 and decays to 21.13 while the anchor's rises monotonically to
31.47, so the anchor also does not robustly overfit where cross-entropy does. §2.3 needs rewriting
around these two cells.

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

## 4. Tiny-ImageNet and WideResNet — the other server

Three tables are empty here and are being filled elsewhere: `tab:tin` (Tiny-ImageNet-200, ResNet-18),
and `tab:wrn`, which carries CIFAR-10 and CIFAR-100 on WideResNet-34-10 side by side. This section is
written to be the whole brief for that machine — everything it needs is committed on
`awp-longschedule-20260730`, so `git pull` and the tables below are sufficient.

### 4.1 Check out this state, not an older one

    git fetch && git checkout awp-longschedule-20260730 && git pull

Two commits on this branch change what these runs produce, and a checkout that predates either one
will silently produce different numbers than the paper reports.

| commit | what it changes | who it affects |
|---|---|---|
| `c5d4458` | `get_model`'s CIFAR-100 branch had **no `arch == "WideResNet"` path** and fell through to ResNet-18. A CIFAR-100 WideResNet run before this commit trains an 11.2M model while its log says WideResNet-34-10 (48.3M). | **every CIFAR-100 WRN cell**, ours and all ten baselines |
| `021f0ef` | the AWP fix, §4.4 below | every cell that runs `train_feat_direction`, i.e. ours on all three tables |

Verify both landed before starting anything, since neither failure announces itself:

    git log --oneline | grep -E "021f0ef|c5d4458"
    python scripts/check_arch.py     # exits nonzero if CIFAR-100 WideResNet builds an 11.2M model

### 4.2 Teachers first — none of the three exist on this machine

Every cell in all three tables reads a naturally trained teacher, and each table needs a different
one. These are the long pole; start all three before anything else.

| checkpoint | config | reported clean | cost |
|---|---|---|---|
| `TinyImageNet/checkpoint/clean_200ep/clean_last.pkl` | `config/TinyImageNet/clean_200ep.yaml` | 66.29 | ≈25 h |
| `CIFAR10/checkpoint/clean_wrn_200ep/clean_last.pkl` | `config/CIFAR10/clean_wrn_200ep.yaml` | — | ≈20 h |
| `CIFAR100/checkpoint/clean_wrn_200ep/clean_last.pkl` | `config/CIFAR100/clean_wrn_200ep.yaml` | — | ≈20 h |

If a machine already holds any of them, copy rather than retrain — but copy the **200-epoch** one.
Tiny-ImageNet has an 80-epoch teacher too (clean 65.97, and its last recorded segment 55.32 → 59.43 →
65.97 shows it had not converged), and the two teachers give materially different students: 57.08 /
18.96 from the 80-epoch teacher against 55.16 / 20.54 from the 200-epoch one. The paper's row is the
200-epoch one.

### 4.3 The champion recipe, exactly

One recipe, no per-dataset tuning — that is a claim the paper makes, so the four configs are
knob-for-knob identical apart from `dataset`, `arch` and the checkpoint paths. Verified 2026-09-02
by diffing all 31 recipe knobs; the only textual differences left are the ones just named, plus
`reformation`, which is dead here (`methods.py:2175` and `utils.py:492` read it **only** as a
fallback when `student_norm` / `teacher_norm` are absent, and all four configs set both explicitly).

| table | config | teacher |
|---|---|---|
| `tab:tin` | `config/TinyImageNet/featdir_tin_champ.yaml` | `TinyImageNet/checkpoint/clean_200ep` |
| `tab:wrn`, CIFAR-10 | `config/CIFAR10/wrn_champ_freezehead.yaml` | `CIFAR10/checkpoint/clean_wrn_200ep` |
| `tab:wrn`, CIFAR-100 | `config/CIFAR100/wrn_champ_freezehead.yaml` | `CIFAR100/checkpoint/clean_wrn_200ep` |
| *(reference)* CIFAR-100 RN18 | `config/CIFAR100/l2_bestrecipe_freezehead.yaml` | `CIFAR100/checkpoint/clean_200ep` |

⚠ **Do not use `config/TinyImageNet/featdir_tin_100ep.yaml` for the headline row.** It is the
80-epoch-teacher cell (57.08 / 18.96) and is reported as the teacher-length ablation. The headline
config is `featdir_tin_champ.yaml`, added 2026-09-02 precisely because the epoch count in the old
name was the student's, not the teacher's, and the difference was invisible.

The payload, so a config can be checked without opening it:

    method: feat_direction        featdir_freeze_head: True     featdir_angeps_p: 1.0
    featdir_rawteacher: True      featdir_rawstudent: True      featdir_span: random
    student_norm: False           teacher_norm: False           gain_head: False
    optim: AdamW   lr: 0.021   cyclic: True   epochs: 100   batch_size: 128   eta: 512
    alpha: 1.0     lamda: 0.0     kappa: 0.999   aug: none
    eps: 8/255     steps: 10      step_size: 2/255/1      train_eps: 0.03450980392156863   # 8.8/255
    weight_avg: True    wa_start: 0.2
    awp_style: proxy    awp_gamma: 0.005    awp_warmup: 10
    load: True     finetune: True    # `finetune` is what warm-starts the student AT the teacher

Four of those are load-bearing and worth re-checking by eye on any config that gets edited:
`featdir_freeze_head: True` (the head is never trained — this is what removed $\tau$ and $\beta$ from
the method), `featdir_angeps_p: 1.0` (sensitivity-matched $\varepsilon$), `train_eps` at 8.8/255
rather than 8/255, and `finetune: True`, which is the warm start. `student_norm: False` with
`teacher_norm: False` is the full-raw pairing; a mismatched pair pins the student's feature norm and
is a different method.

### 4.4 What the AWP fix changed, and why the numbers barely move

`_awp_loss_fn` — the closure AWP ascends — diverged from `_step_loss`, the closure the model
descends, in two ways. It had no `freeze_head` branch, so it kept a head KL term that
`freeze_head: True` deletes from training; and the `alpha` detach was applied to the normalized
feature while the raw branch feeds the unnormalized one. AWP was therefore ascending on an objective
the model does not train. Measured before the fix: the head term was 7.15 against the anchor's 4.15
and supplied **40.8% of the AWP backbone gradient**, at cosine 0.918 to the correct direction.
Separately, `freeze_head` was applied *after* the AWP proxy was deepcopied, which is only safe
because `awp_warmup: 10` delays the first proxy step.

Both are fixed in `021f0ef`. The re-measurement:

| | before | after | ΔNRR |
|---|---|---|---|
| CIFAR-100, `l2_bestrecipe_freezehead` | 62.65 / 28.77 / 39.43 | **62.17 / 28.86 / 39.42** | 0.01 |
| CIFAR-10, `champ_eps88` | 85.58 / 51.79 / 64.53 | **84.96 / 51.74 / 64.31** | 0.22 |

So AWP is insensitive to the 41% of its ascent direction that was wrong, and **no result changes its
sign or ordering**. The fix still has to be in the checkout, because the paper describes the code and
the code now matches the description — not because the other server would otherwise get a different
answer. Its practical consequence for that machine is the opposite of a hazard: it means a cell
already started on an older checkout does **not** need restarting.

### 4.5 The baseline code is on this branch

All ten baseline objectives are implemented in `methods.py` here and dispatch through
`getattr(methods, "train_" + config.method)`. Nothing needs to be ported from `../IGDM`, and the
natural-teacher versions are the ones committed — that is the point of running them ourselves.

| row | `method:` | `methods.py` | notes |
|---|---|---|---|
| PGD-AT | `madry_at` | `train_madry_at` | `finetune: False` → random init |
| TRADES | `trades` | `train_trades`, `_trades_inner_attack` | $\beta = 6$ as published |
| MART | `mart` | `train_mart` | $\lambda = 5$ as published |
| PGD-AT @ teacher init | `madry_at` | `train_madry_at` | same code, `finetune: True` |
| ARD | `ard` | `train_ard` | `_DistillStack`, `_kd_inner_attack` |
| RSLAD | `rslad` | `train_rslad` | `_DistillStack`, `_kd_inner_attack` |
| AdaAD | `adaad` | `train_adaad` | `_adaad_inner_attack` |
| AdaAD + IGDM | `adaad_igdm` | `train_adaad_igdm` | `_adaad_inner_attack` |
| CFA (ours) | `feat_direction` | `train_feat_direction` | + `utils.inner_featdir_teacher_at_adv` |
| ADR, CURE, RPAT++ | — | external repos | not on this branch; see §5 |

Configs are complete and audited (2026-09-02): 8 baselines × 3 tables, plus 3 teachers and 3
champions. Tiny-ImageNet drops the suffix, both WideResNet sets take `_wrn`:

    pgdat_100ep  trades_100ep  mart_100ep  at_teacherinit_matched
    ard_nat100ep  rslad_nat100ep  adaad_nat100ep  adaadigdm_nat100ep

The three PGD-AT/TRADES/MART configs carry a `checkpoint:` line but `finetune: False`, so they load a
teacher for nothing and train from random init, which is correct — `at_teacherinit_matched` is the
only one of the four with `finetune: True`. Do not "fix" the apparent redundancy.

### 4.6 Driver, order and cost

    bash scripts/run_tin_wrn_fill.sh TIN core      # 3 baselines, then add ours by hand
    bash scripts/run_tin_wrn_fill.sh WRN10 core
    bash scripts/run_tin_wrn_fill.sh WRN100 core
    bash scripts/run_tin_wrn_fill.sh TIN           # full 8

`core` is PGD-AT, AdaAD and PGD-AT-at-teacher-init. Those three plus ours carry every claim the
tables are used for: the reference point, the strongest distillation objective on a natural teacher,
the "do not distil at all" bound that all four distillation rows fall below, and ours. Twelve cells
rather than 33, ≈104 GPU-hours rather than ≈329. The driver refuses to start if its teacher is
missing. **It does not run the champion** — start that separately, alongside:

    python main.py --config_name featdir_tin_champ.yaml   --dataset TinyImageNet --seed 0
    python main.py --config_name wrn_champ_freezehead.yaml --dataset CIFAR10     --seed 0
    python main.py --config_name wrn_champ_freezehead.yaml --dataset CIFAR100    --seed 0

Per-cell cost against CIFAR/ResNet-18: Tiny-ImageNet ×4.0 (64×64, 100k images), WideResNet-34-10
×4.5 (48.3M against 11.2M). `wrn_champ_freezehead` sets `aa_batch_size: 256`; on WSL2 AutoAttack is
kernel-launch bound rather than VRAM bound and the default batch made a WideResNet evaluation run
12× slower than it should.

Do not run two trainings on one GPU. Three collisions have already been caused by independent
`while pgrep; sleep` waiters; use one sequential driver per GPU.

### 4.7 Still owed by this machine

The Tiny-ImageNet log for 55.16 / 20.54 lives on the other server, so `results/TinyImageNet/` here
has no AutoAttack line for it. **Copy that log back**, or the paper's number cannot be regenerated
from this repository.

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
