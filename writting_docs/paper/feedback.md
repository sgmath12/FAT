# Writing feedback

Standing feedback on how the paper should be written, recorded so it does not have to be given
twice. Newest first. Each entry says the rule, why it exists, and what triggered it, because a rule
without its trigger tends to get re-broken in a new place.

---

## F2 — Table captions are not where experimental detail goes (2026-09-05)

> "누가 테이블 캡션을 그렇게 길게 써. 그런건 어펜딕스에 뭐 실험 디테일한 세팅. 이렇게해서 적어야지"

**Rule.** A caption says what the table shows and under what setting, in one or two sentences.
Everything else — which variant of a baseline was run, why a hyperparameter was chosen, what a
number should be read against, the history of a constant — goes in appendix prose.

**Why.** A reader scanning a table wants to know what they are looking at. A caption that carries
six per-method footnotes is unreadable in the position it appears in, and it hides the detail from
anyone reading the appendix, where they would actually look for it.

**Trigger.** `tab:main`'s caption had grown to 102 words plus seven lettered footnotes (j–o, ‡)
covering LBGAT's variant, HAT's schedule, ADR's configuration, IGDM's alpha and the logit anchor's
temperature. `tab:published_tin` was 225 words. Fixed by cutting captions to a few sentences and
creating `\section{Notes on the Baselines}` (`app:baselines`) for the per-method decisions.

**How to check.** Any caption over about 60 words is suspect. `grep`-count the words inside
`\caption{...}` before committing.

---

## F1 — Tables belong on the page that discusses them (2026-09-05)

> "메인 페이지에서 테이블 실험테이블을 싹다 어펜딕스로 보내면 어떡하니"

**Rule.** After adding or growing a table, check the page it actually lands on in the built PDF,
not the line it is declared on in the source.

**Why.** LaTeX floats silently. A table declared in the experiments section can be deferred past
the end of the document and land in the appendix, and nothing in the build output says so.

**Trigger.** Seven tables in `4_Experiments.tex` were all declared `[t]`. With only top placement
allowed and a 13-page body, the queue overflowed: `tab:ladder` and `tab:controls` landed on page 25,
`tab:tin` on page 24. Fixed with `\usepackage[section]{placeins}`, wider float parameters
(`topnumber` 3, `totalnumber` 4, `topfraction` 0.92, `textfraction` 0.06) and `[!htbp]` throughout.

**How to check.**

```
python3 - <<'PY'
import pypdf
r = pypdf.PdfReader('writting_docs/paper/main.pdf')
pages = [p.extract_text() or "" for p in r.pages]
for name, needle in {'tab:ladder': 'Component ablation'}.items():
    print(name, [i+1 for i, t in enumerate(pages) if needle in t])
PY
```


## Abstract

Adversarial training is among the most reliable defenses against adversarial examples, but it degrades
standard accuracy, and this trade-off is the central obstacle to its use.
[feedback] `콤마가 일단 두 개 들어간거 괜찮은거임 ? 그리고, 마지막에 and this trade-off is the central obstacal to its use 여기서, use라는 단어가 학술에 어울릴지랑, 이게 굳이 central obstacle 까지인가 싶어. 좀 더 낮춰서 표현안되나`

The accuracy is not lost in general: a network of the same architecture trained naturally on the same data still holds it.
[feedback] `이 문장 자체가 너무 구림.. is not lost in general 이것도 문장 자체가 좀 구어체같고, 그리고 의미도 이거 너무 당연한 거 아님 ? `

We ask whether self-distillation can put it back, leaving robustness where it already is, in the inner
maximization.
[feedback] `갑자기 난데없이 self-distillation이 왜나옴? 빌드업이 전혀안되어있음 `


Such a teacher is the extreme case of what adversarial distillation says goes wrong---the
field diagnoses the target as too sharp and builds machinery to soften it, and a naturally trained
network's maximum softmax probability is $0.820$ against an adversarially trained student's $0.016$.
Softening does help, raising AutoAttack accuracy from $20.84$ to $24.48$ across a temperature sweep,
and it leaves standard accuracy flat at $58$ while the teacher holds $77.66$: it repairs robustness,
not the axis we came for. Reading the same teacher's \emph{feature} instead recovers both.

[feedback] `이거는 너무 숫자가 덕지덕지 나옴. 야 .abstraction인데 이렇게 이런 측정값을 막 집어넣음 어떡해 최종 성능도 아니고. 이렇게 쓰면 안돼  `


We anchor the student's adversarial feature to the teacher's clean feature, so the teacher is never evaluated
under attack and transfers neither its instability nor any robustness, and prove that this single term
is equivalent within a constant factor to fidelity plus local stability, with no coefficient between
them.

[feedback] `We anchor the student's adversarial feature to the teacher's clean feature 까지는 좋음. 근데 그 이후가 갑자기 뭐? 티처가 뭐 never evaluate 이런말을 왜함 ? 문장 좀 다시써봐  `

Combined with a per-sample attack radius set by the input sensitivity of the loss at fixed total
budget, the method has no loss weight, no temperature and no trained classifier.

[feedback] `has no loss wegith 이것도 뭔가 너무 구려. 제발.  `

The anchor is not
merely compatible with robustness: with no weight averaging, no AWP and no radius rule, it leads label
cross-entropy by $5.91$ points of AutoAttack and $6.69$ of standard accuracy under an identical
schedule. At a fixed weight-averaging and AWP stack it matches the best published AutoAttack accuracy
on CIFAR-100 to within $0.27$ points while keeping $5.29$ more points of standard accuracy, and
transfers unchanged to CIFAR-10 and Tiny-ImageNet.


[feedback] `전반적으로, ./REFERENCE  폴더의 IGDm, long-tailed-at  앱스트랙션 보고 그거랑 비슷한 흐름 및 문체 및 단어 선택. 으로 좀 써봐 `

---

## F3 — The abstract, rewritten (2026-09-05)

Seven separate notes on the abstract, and what each one changed. The rewritten version is in
`0_Abstract.tex`; the version they were written against is in git at `de02086`.

| # | Feedback | What changed |
|---|---|---|
| 1 | Two commas in the opening sentence; is `use` the right register; is `central obstacle` overclaiming | Opening is now `Adversarial training improves adversarial robustness, but it does so at a cost in standard accuracy.` One comma, no `central obstacle`, no `use` |
| 2 | `The accuracy is not lost in general` reads colloquial, and says something obvious | Sentence deleted outright |
| 3 | `self-distillation` appears with no build-up | The word is gone. The build-up is now explicit: distillation reduces the cost with a robust teacher, that teacher costs more than the student, `In contrast to these approaches` we use a natural one |
| 4 | Measurements scattered through the abstract that are not even final performance | `0.820`, `0.016`, `20.84`, `24.48`, `58`, `77.66` all removed. The only numbers left are `62.17 / 28.86` in the last sentence |
| 5 | `so the teacher is never evaluated under attack ...` — why is this here | Clause deleted; the sentence ends at `anchors the student's adversarial feature to the teacher's clean feature` |
| 6 | `has no loss weight` reads badly | `has` to `requires` |
| 7 | Follow the flow, register and word choice of IGDM and long-tailed-at | Structure copied from them: setup, what existing work does, `In contrast to these approaches`, `In this paper we propose`, `Experimental results show that`. Numbers only in the final sentence, as in IGDM |

**Rules taken from this, for everything else in the paper.**

1. **An abstract carries final performance and nothing else numeric.** Intermediate measurements --
   temperature sweeps, softmax probabilities, ablation deltas -- belong in the sections that
   establish them. If a number is not the result being claimed, it does not go in the abstract.
2. **Nothing appears without being introduced.** A method name, a technique, a design choice: the
   sentence before it has to make the reader expect it.
3. **State the contrast where the reader is standing.** `In contrast to these approaches` after
   describing those approaches, not a claim about ourselves that the reader cannot yet place.
4. **Do not explain a property before the claim it supports.** The abstract says what the anchor
   is; why the teacher is never attacked is a section, not a clause.
5. **Lower the register on setup claims.** `at a cost in standard accuracy` rather than `the central
   obstacle to its use`. The result should be the strongest sentence in the abstract.
6. **Reference style is the two papers named above**, not this repository's own prose habits: short
   declarative sentences, no em-dash asides, plain verbs.

**Still to apply.** Points 1-6 were written about the abstract but read as general. The introduction
and section openings have not been reread against them.

### F3b — three more passes on the same abstract

The first rewrite was rejected twice more. What was still wrong, and the rule each time.

**Pass 2 --- `but it does so at a cost in standard accuracy` is weak, and adversarial distillation
arrives unconnected.** The reference abstracts were read for structure and not for sentence
construction. Their second sentence is *derived from* the first --- IGDM: "superior performance is
primarily attained with large models. **This substantial performance gap** ... has spurred active
research into AD" --- where mine named a new field with no bridge. Rules: **do not open a clause with
`it`**; **every sentence after the first has to attach to the one before it**, and if it cannot, the
first sentence is the one that is wrong.

**Pass 3 --- adversarial distillation does not belong in the abstract at all.**

> "굳이 AD를 끌어들일 필요가 없어 지금은. 차라리 내츄럴 모델을 사용하는 애들을 끌고와야지."
> "AD는 인트로에 좀 적고 related work 및 메소드에서 비교하면 돼"

The abstract's contrast has to be with the work our contribution actually differs from --- the methods
that already use a naturally trained network (LBGAT, ARREST, DP-FAT, B-MTARD) --- not with the
literature we merely sit near. Adversarial distillation belongs in the introduction and in the
comparisons, where there is room to say what it is. **Rule: the abstract names one line of prior work,
and it is the one the contribution is defined against.** The difference is now stated as what it is:
they add the natural network beside the label loss and tune the weight, we remove the label loss.

An error was also caught in pass 2's draft. It claimed the accuracy loss motivated adversarial
distillation. It did not --- AD exists because small models are hard to make robust. **Rule: a
narrative bridge is still a factual claim.**

**Pass 4 --- word choice, pronouns, commas, one at a time.** A sentence-level audit found: `models
sacrifice` (models are not agents), `leverage X, and depend on Y` (a comma before `and` joining two
predicates of one subject), a relative clause splitting `guide ... and transfer`, two `and`s in one
sentence, `sacrifice` repeated from the opening, and a closing sentence carrying four commas.

**Checks to run on any paragraph before calling it done.**

| check | threshold |
|---|---|
| words per sentence | over 28 is suspect |
| commas per sentence | over 1 needs a reason; over 2 is a rewrite |
| comma before `and` | only when joining independent clauses |
| sentence-initial `it` / `this` / `these` | none, unless followed by a noun (`This gap`) |
| repeated content word | not twice in a paragraph |
| abstract length | 150--200 words, against IGDM's 150 |

Result: 337 words and 6 embedded measurements, down to 224 words and one pair of final numbers.
