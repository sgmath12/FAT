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


Adversarial training significantly improves adversarial robustness, yet robust models attain
substantially lower standard accuracy than naturally trained ones.
A naturally trained network of the same architecture retains that accuracy and costs nothing adversarial to obtain.
>> 굳이 ? 첫째로, 이건 너무 당연한 거고 이미 문제점은 언급함 (로버스트 모델은 lower standard accuracy). 둘재로,  costs nothing adversarial to obtain. 이게 전혀 학술적이지 않아 . 일단 nothing adversarial 이라는 게 너무 애매해

Existing methods add such a network to adversarial training as an auxiliary target beside the label loss, and the weight
between them has to be tuned.

>> such a network 가 뭐니 애매하잖아. 그냥 existing mehthods adds a addtional regularization to loss to enhance clean accuracy or utilize natural model's logit to enhancw ~~~ 뭐 이런식으로 해야지.


In contrast to these approaches, we remove the label loss and make the natural network the sole target of the backbone.

>> 우리도 비슷하게 내츄럴 모델을 이용하는데, 우리는 좀 더 간단하지만 성능적으론 그렇지 않은, 혹은 더 내츄럴모델을 훨씬 더 잘이용하는 방법을 이용한다. 뭐 이렇게 해야지.

In this paper, we propose Clean Feature Anchoring (CFA), which anchors the student's adversarial
feature to the teacher's clean feature.



The anchor is read one layer before the softmax, because a natural teacher's logits do not carry the accuracy they encode.
>> 여기도 좀 이상해. beacuase 이렇게 시작하는게 다소 논문스럽지않고, one layer before the softmax라 할 필요가없어 그냥 feature라 하면 돼. 저게 유일한것도아니고.  그리고 티처 의 로짓이 do not carry eht accuracy they eocnde ㅇ이것도 좀 추상적이야.

We prove that this single term controls fidelity to the teacher and local stability of the student within a constant factor, so no coefficient balances them.
>> 문장 자체가 좀 단어도 그렇고 학술적인 느낌이 아니야. 뭐랄까 cvpr iclr 에 이런문장 안나올거같아.

We further set the per-sample attack radius from the input sensitivity of
the loss under a fixed total budget.
>> 왜 추가했어? under a fixe total budget to utilize ~ to enhance ~ to 극대화하다~ to 우리가 제안한 로스를 좀 더 최적화햇따~ 등

The resulting objective requires no temperature, no loss weight and no trained classifier.

>> 이건 일단 좀 약해. 굳이야. 

Experimental results show that CFA recovers standard accuracy without giving up robustness.
>> giving up 이게 오바잖아 without sacrifying 이라하던지

Particularly, CFA attains $62.17\%$ standard accuracy at $28.86\%$ AutoAttack accuracy on CIFAR-100
with ResNet-18 and transfers unchanged to CIFAR-10 and Tiny-ImageNet.

>>ㅇ이것도 갑자기 잘 쓰다가 뭔 transfer uncahnged to 이게뭐야 대체

No published method is ahead on both axes.

>> 너 진심 이런 문장이 학회 앱스트랙션에 나올거같냐. 정신차려라.


### F3c --- pass 5, the eight inline notes on the fourth draft

| # | Feedback | What changed |
|---|---|---|
| 1 | `A naturally trained network ... costs nothing adversarial to obtain` --- obvious, and `nothing adversarial` is not a phrase | Sentence deleted. The problem was already stated by the opening |
| 2 | `such a network` is vague; say what existing methods actually do | Replaced by two sentences: the gap motivated bringing a natural network in, and `Existing methods distill its logits into the robust model or add its predictions as an auxiliary regularizer beside the label loss` |
| 3 | The contrast should be *we use it too, but better*, not *we are different* | `We likewise employ a naturally trained network, but we exploit it far more directly` |
| 4 | `read one layer before the softmax, because ...` --- do not open with `because`, say `feature`, and the logit claim is vague | Whole clause dropped; the contrast is now inside the CFA sentence, `rather than to its logits` |
| 5 | The proposition sentence does not read like a paper | `bounds both fidelity to the teacher and local stability of the student up to a constant factor, so that no balancing coefficient is required` |
| 6 | The radius sentence never says what it is for | Purpose clause added: `which concentrates the anchor where the feature is most fragile` |
| 7 | `requires no temperature, no loss weight and no trained classifier` is weak | Deleted. Simplicity is already carried by `the label loss is discarded entirely` |
| 8 | `without giving up robustness` | `without sacrificing robustness` |
| 9 | `transfers unchanged to CIFAR-10 and Tiny-ImageNet` | `consistent improvements are observed on CIFAR-10 and Tiny-ImageNet`, following IGDM's closing form |
| 10 | `No published method is ahead on both axes.` does not belong in a conference abstract | Deleted |

**Rules added.**

7. **Position the contribution inside the line of work it belongs to, not outside it.** If we use the
   same ingredient as prior work, say so and claim the better use of it. `In contrast to these
   approaches` is for a different ingredient (IGDM: logits vs.\ input gradient); it is the wrong
   frame when the ingredient is shared.
8. **Every design choice in an abstract carries its purpose in the same sentence.** A sentence that
   only says what was done invites the question the reader then asks of the whole method.
9. **A negative list is not a contribution.** `no temperature, no loss weight, no classifier` states
   absence; the positive form (`the label loss is discarded entirely`) says the same thing as a
   decision.
10. **Nothing rhetorical in the last sentence.** The final sentence is the headline number and where
    else it holds, in the reference abstracts' form.

Result: 212 words, longest sentence 31 words, no sentence over 2 commas.

**Pass 6 --- one word.** `We likewise employ` --- `likewise` is not a word that appears in this
literature. Changed to `We also employ`. **Rule: if a connective cannot be recalled from a paper in
the field, it is the wrong connective, however correct it is.**

---

## F4 --- The introduction, rewritten under the abstract's rules (2026-09-05)

> "지금 인트로 메소드 모두 다 약간 처음 앱스트랙션처럼 개판이야."
> "철저히 논리적 구조 따르면서 접속사 선택, 너무 짧은 문장 지양, 너무 많은 콤마 사용 지양, 왜 이런 걸 하는지에 대한 근거 혹은 목적."

**The pattern behind every note so far.** The prose was written as an essay rather than as a paper:
rhythm instead of structure, a short rhetorical sentence after a long one, asides addressed to the
reader, and evidence placed where the eye falls (abstract, captions) rather than in the section that
establishes it.

**What was fixed in `1_Intro.tex`.**

| Symptom | Instances removed |
|---|---|
| rhetorical fragment | `Its cost, however, is standard accuracy.` / `The two lines leave a gap between them.` / `It survives one layer earlier, and that is where we put the anchor.` / `Two consequences follow from the form alone.` |
| overclaimed setup | `remains the most reliable` (now `most reliable under strong evaluation`), `the principal obstacle to deploying` (now `a major limitation`) |
| aside to the reader | `drawn for reading rather than fitted for a claim`, `so it is ahead on both axes rather than trading one for the other`, `no method in this setting exceeds ours on both axes` |
| caption carrying the results | `fig:frontier` cut from 131 words with six number pairs to 54 words with none (F2 had been recorded and then broken in a figure) |
| intermediate measurements | `0.820`, `0.016`, `20.84`, `24.48`, `58`, `77.66`, and the three AD-on-natural-teacher numbers, all deferred to \Cref{sec:analysis} |
| wrong position in the literature | `Three lines of work come close` (ARREST, DP-FAT, B-MTARD as near misses) becomes `We also employ a naturally trained network, but we exploit it far more directly`, matching the abstract |
| purpose missing | the radius paragraph now opens with why (`in order to equalize what the attack accomplishes on the quantity the objective actually measures`) before what |

**Sentence budget after the pass.** 1154 words. Four sentences over 32 words remain, and no sentence
carries more than two commas outside a citation list. The longest sentences are the two that state
the bound and the two that state the headline numbers, which is where length is earned.

**Rule 11.** A paragraph states its purpose before its mechanism. If the first sentence of a
paragraph says what was done, the reader has to hold it unexplained until the paragraph ends.

**Rule 12.** No sentence exists to set up the next one. Every rhetorical fragment in the list above
was there to make the following sentence land, and each one was deletable without losing content.

