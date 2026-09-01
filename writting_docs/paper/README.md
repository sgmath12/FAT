# Paper source

Target venue **ICLR 2027**. Build with `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

```
main.tex          preamble + \input of the sections; the only file that names the venue kit
0_Abstract.tex    1_Intro.tex   2_Analysis.tex   3_Method.tex   4_Experiments.tex   5_Appendix.tex
AT.bib            bibliography reused from the IGDM ICLR'25 release, so shared works keep their keys
cfa.bib           19 entries AT.bib lacks (ARREST, LBGAT, CURE, ADR, RPAT, B-MTARD, Generalist++,
                  DP-FAT, IGDM, IAAT/MMA/CAT, Ilyas et al., the ICML'26 teacher-quality paper)
style/            venue kit -- swap for the ICLR 2027 .sty/.bst when it is released
notes/            prose companions to each section, carrying the sourcing of every number and the
                  open decisions behind it.  NOT part of the build.
```

**Conventions**, taken from the reference kit rather than invented: `\Cref` for every cross-reference,
`\epsilon` (not `\varepsilon`), the `eqn:` / `tab:` / `sec:` / `app:` label namespaces, and AT.bib's
citation keys wherever the work is shared, so the two bibliographies compose without duplicates.

`\iclrfinalcopy` is commented out and the author block is the double-blind placeholder. Uncomment
only for camera-ready.

**Cells marked `\dg`** in `4_Experiments.tex` are being re-measured under the corrected AWP ascent
objective (`methods.py`, commit `021f0ef`). Cells without AWP never enter that code path and are final.
