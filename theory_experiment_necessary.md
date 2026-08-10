# Experiments the theory section needs (2026-08-09)

Plan of record for closing theory_v1.md's open items. Predictions are **registered here before the
runs**; decision rules are written so that every outcome has a pre-agreed reading. Results get
appended to §Status at the bottom — do not edit the predictions after the fact.

**Why this plan exists.** The champion-regime evidence for the directional design carries three
auxiliary devices (WA, AWP, $\varepsilon_{\mathrm{tr}}=8.8/255$, frozen-LR tail), and the 50ep
no-stack grid *inverted* the normalization ordering (raw won PGD at tied clean). So it is currently
unknown whether the clean advantage of the directional target is **design-intrinsic** or an
**interaction with the stack**. The paper's main ablation must run on a clean base; the stack goes
in a separate bridge table. The champion-regime mechanism cells (`npen01`, champion `tnormsraw`)
were canceled mid-run on 2026-08-09 in favour of this plan.

---

## Phase 0 — the "paper-base" regime (fixed, do not tune per cell)

ResNet-18 / CIFAR-100, teacher `clean_200ep` (77.59), student init = teacher (backbone+head).

| knob | value | note |
|---|---|---|
| epochs | 100 | 50ep showed the inversion; 100 is the paper base |
| optim / lr | AdamW 0.021, plain one-cycle | **no freeze_lr** |
| attack | PGD-10, $\alpha=2/255$, $\varepsilon_{\mathrm{tr}}=8/255$ | **exactly eval radius, no 1.1×** |
| WA / AWP / angeps | **all off** | the whole point |
| loss | $k=d$, $\lambda=0$, $\beta=1$, $\tau=16$, head input normalized (`student_norm: True`) | |
| eval | `last` checkpoint, clean / PGD-20 / CW / **AA** | AA on every cell — PGD/CW margins are known not to transfer (project rule: only AA arbitrates) |
| seed | 0 (phase 4 adds seeds on the key pair) | |

## ⚠ ERRATUM (2026-08-10): Phase 1's "raw" cells are PARTIAL raw

Every Phase 1 cell keeps `student_norm: True`, so the head input **and the eval-time forward** are
normalized: only the backbone target is raw. That is the `rawfeat` variant, not the full raw design
(`student_norm: False`, where $\lVert\Phi_s\rVert$ becomes the logit scale). The two differ
materially — at the champion regime, partial raw lands $59.79$/AA $28.57$ and full raw
$57.78$/AA $28.04$. **The same confusion already occurred once** (the `fg_*` grid of 2026-08-01
toggled normalization inside the direction loss only; the `n2_*` grid was built to fix it), and
Phase 1 reverted to the pre-fix form. Consequence: **P-C is established for partial raw only.**
Full-raw evidence that exists: 50ep no-stack `n2_traw_sraw` $62.86/32.36/27.18$ (AA never run,
checkpoint on disk) and champion-regime `fullraw` above. Missing: full raw at 100ep no-stack, and
full raw on CIFAR-10. Label every future cell partial/full in its config header.

## Phase 1 — the normalization 2×2 at paper-base (RUNNING)

Four cells, varying only `featdir_rawstudent` / `featdir_rawteacher` (the direction-loss sides;
head input stays normalized in all four):

| config | loss | algebra says |
|---|---|---|
| `b2x2_snorm_tnorm` | $\lVert\hat s-\hat t\rVert^2=2-2\cos\theta$ | direction only |
| `b2x2_snorm_traw` | $1+\lVert\Phi_t\rVert^2-2\lVert\Phi_t\rVert\cos\theta$ | direction × per-sample weight $\lVert\Phi_t\rVert$ |
| `b2x2_sraw_tnorm` | $\lVert\Phi_s\rVert^2-2\lVert\Phi_s\rVert\cos\theta+1$ | + magnitude command $\lVert\Phi_s\rVert\to\cos\theta$ |
| `b2x2_sraw_traw` | $\lVert\Phi_s-\Phi_t\rVert^2$ | + magnitude command $\lVert\Phi_s\rVert\to\lVert\Phi_t\rVert\cos\theta$ |

Run order: `snorm_tnorm` → `sraw_traw` (headline pair first) → `snorm_traw` → `sraw_tnorm`.

**Registered predictions (from theory_v1 T.4's 2×2 expansion):**

- P-A (design-intrinsic): top row ties on all metrics; bottom row pays **clean** at an AA tie, with
  $\lVert\Phi_s\rVert$ crushed toward the target scale ($\approx\cos\theta$ for `sraw_tnorm`,
  $\approx\lVert\Phi_t\rVert$ for `sraw_traw`). → T.4's claim stands at the clean base; champion
  stack becomes a booster, not a precondition.
- P-B (stack interaction): all four tie (the 50ep pattern persists at 100ep). → the directional
  clean advantage **emerges only under the stack**; T.4 must be rewritten as a conditional claim
  and the interaction itself becomes a finding. Do not hide this outcome.
- P-C (inversion): raw wins something at AA. → the champion-regime $-0.95$ was itself
  stack-induced; major revision of T.3/T.4.

**Decision rule:** whichever of P-A/P-B/P-C matches AA (not PGD), that reading goes in the paper.

## Phase 2 — mechanism cell, conditional on P-A (+1 run)

`b2x2_snorm_tnorm` + `featdir_norm_penalty` (knob already implemented in methods.py): pure cos
gradient + explicit $\mu(\lVert\Phi_s\rVert-\lVert\Phi_t\rVert)^2$, $\mu=0.1$ first.

- clean drops → binding the norm **per se** is the cost (capacity/selection account).
- clean holds → the cost was the raw gradient's **geometry** (radial leakage / per-sample
  weighting — the dynamics account). Follow-up: per-sample dir-gradient stats on the Phase 1
  checkpoints (no training; CV, radial:tangential ratio, weight-vs-norm correlation).

Skip entirely under P-B/P-C (no phenomenon at base to explain).

## Phase 3 — bridge table + completion

1. Winning cell + stack, one lever at a time (+WA, +AWP, +8.8, +angeps) — mostly exists from the
   champion line; fill gaps only.
2. Seeds ×3 on the key pair (`snorm_tnorm` vs `sraw_traw`).

## Backlog — other theory predictions still owed (priority order)

1. **Prediction 1b (robust-teacher swap ≈ 0)** — same-architecture robust teacher
   (`CIFAR100/checkpoint/at_ce_freehead/madry_at_last.pkl`, clean-init AT, shares basis) as
   teacher+init at paper-base. **Must run before submission**: if the robust teacher wins
   materially, Corollary 1's scoped claim also falls — better to know first.
2. **cos-clean scatter** — $\cos(\hat\Phi_s(x),\hat\Phi_t(x))$ vs clean over ALL champion-family
   checkpoints on disk (no training; extends T.3's 3-point monotonicity to 10+ points).
3. **Prediction 2 (head-freeze cost grows with $\varepsilon_{\mathrm{tr}}$)** — freeze ×
   $\varepsilon_{\mathrm{tr}}$ grid, 3 points, paper-base.
4. Per-sample gradient statistics script (only if Phase 2 lands on the dynamics branch).
5. **Attack-action decomposition** — ✅ **MEASURED 2026-08-09** (`measure_attack_action.py`,
   PGD-CE, $n=1024$). Registered prediction ("relative rotation exceeds relative shrinkage"):
   **REJECTED as worded.** Teacher: 62° rotation but norm **inflates ×2.47** — norm damage (1.47)
   exceeds rotation damage (0.54); the DGP inequality $\cos<$ ratio itself holds (99.9 %), driven
   by inflation. AT students: 12–13°, ~5 % norm change, inequality only ~41 %. Consequence:
   "the attack mainly hits the angle" is unusable as a premise; what survives is "the natural
   feature's norm is its most attack-volatile, class-empty coordinate" (theory_v1 T.4 positive
   premise, rewritten post-measurement).

---

## Status

- 2026-08-09: plan created. Champion-regime `npen01` (mid-run) and queued champion `tnormsraw`
  **canceled**; Phase 1 queue launched (snorm_tnorm → sraw_traw → snorm_traw → sraw_tnorm),
  AA on, seed 0.
- 2026-08-09 15:05 — **Phase 1 diagonal complete. VERDICT: P-C (inversion).**
  `snorm_tnorm` 61.52 / 26.92 / 24.48 / **AA 22.90**; `sraw_traw` 62.54 / 29.05 / 26.15 /
  **AA 24.29**. Raw wins **both axes** at paper-base (AA $+1.39$, clean $+1.02$ — outside the
  $\pm0.3$–$0.4$ noise floor, consistent across every metric). Consequences per the registered
  rule: the champion-regime directional advantage (clean $+0.95$ at AA tie) is **stack-induced**;
  T.3/T.4 go under revision as method-plus-stack claims; Phase 2 (npen at base) is **moot**.
  Stack contribution measured at AA: direction $+5.79$, raw $+4.28$ — the loss×stack interaction
  is now the mechanism question (which element flips the ordering is unmeasured; WA is the prime
  suspect and the cheapest next probe: direction+WA vs raw+WA, 2 runs). Both base cells show
  robust overfitting vs their 50ep counterparts (PGD $-2.2$ / $-3.3$). Cells 3–4 still queued —
  they now test whether the *base* ordering follows the student-side split in reverse.
- 2026-08-09 17:10 — **cell 3 (`snorm_traw`) completed before the queue was stopped**:
  $61.28$ / $26.83$ / $24.68$ / **AA 22.98** — ties `snorm_tnorm` ($61.52$/AA $22.90$): the
  **top-row tie holds at base too**, confirming the 2×2 expansion (teacher side = weighting
  detail) in a second regime. Cell 4 (`sraw_tnorm`) killed mid-run per user pivot.
- 2026-08-09 17:15 — **pivot to 50ep** (user call: 100ep-no-stack is robust-overfit, 50ep was the
  regime where no-stack looked healthy). Discovery: the head-fixed 50ep 2×2 already exists as
  checkpoints — `fg_plain_{th_sh,tr_sh,th_sr,tr_sr}_kl` (2026-08-01 grid; NOT the n2 grid, which
  moves the head too). Known 50ep clean/PGD/CW: th_sh $62.61/29.16/26.63$, tr_sr $62.68/30.51/27.24$.
  **AA was never run on them** — AA-eval of all four launched (`aa_eval_fg2x2.py`, no training).
  Question: does the P-C inversion hold at 50ep where neither cell is overfit, or was it
  overfitting-mediated?
- 2026-08-09 17:30 — **50ep 2×2 AA measured** (12 min, bs 256): th_sh (dir) **24.66**,
  tr_sr (raw) **25.14**, tr_sh (weighting-only) **24.76**, th_sr (magnitude-command) **16.29**
  (the known norm-crush collapse, $\lVert\Phi_s\rVert\to0.60$, clean 39.31 — bottom-left cell
  pathology as the algebra predicts). Readings: **(1)** the inversion is *not* overfitting-mediated
  — raw already leads at 50ep ($+0.48$, marginal vs noise but same sign as 100ep's $+1.39$);
  **(2)** the direction loss **robust-overfits harder** without the stack: 50→100ep AA drop
  $-1.76$ (dir) vs $-0.85$ (raw); **(3)** top-row tie again (24.66 vs 24.76). Three-regime
  summary: no-stack raw ≥ dir (gap grows with epochs); stack flips it (dir clean $+0.95$ at AA
  tie). **The stack (= overfitting control: WA averaging + AWP flatness) is what converts the
  directional loss from worse-than-raw into the clean/robust winner** — this interaction is now
  the paper's mechanism object.
- 2026-08-09 ~18:00 — **CIFAR-10 cross-check launched**: `c10_th_sh_kl` (dir) → `c10_tr_sr_kl`
  (raw), the 2026-08-02 prepared-but-never-run 50ep no-stack diagonal, now with `aa: True` +
  `aa_batch_size: 256`. Question: does the no-stack raw≥dir ordering replicate on CIFAR-10
  (loss property) or not (CIFAR-100 particularity)? C100 50ep reference: dir AA 24.66 vs raw
  25.14 ($+0.48$).
- 2026-08-09 22:01 — **CIFAR-10 replicates the inversion.** 50ep no-stack: dir
  $85.79/52.26/50.52$/AA $48.20$ vs raw $86.00/52.24/50.94$/AA $48.73$ — raw $+0.53$ AA,
  nearly the same margin as C100's $+0.48$. **The no-stack raw ≥ dir ordering is a property of
  the loss, not of CIFAR-100.** Remaining unmeasured corner: C10 raw+stack (whether the stack
  flip also replicates on C10 — the dir+stack cell exists, $82.52$/AA $51.89$).
