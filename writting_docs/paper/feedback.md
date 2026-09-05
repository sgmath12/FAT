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
