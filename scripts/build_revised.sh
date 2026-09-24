#!/usr/bin/env bash
# 2026-09-24.  Builds the revised manuscript: main_totel_revised.tex (body) plus appendix_revised.tex.
# The body's \input names appendix.tex, which does not exist, so this writes main_revised.tex with that
# one line pointed at the real file and compiles that.  Output: writting_docs/paper/main_revised.pdf.
set -u
cd "$(dirname "$0")/../writting_docs/paper"
PAPER="$PWD"
OUT="${1:-/tmp/fat_revised_build}"
mkdir -p "$OUT"
sed 's|\\input{appendix\.tex}|\\input{appendix_revised.tex}|' main_totel_revised.tex > main_revised.tex
~/.local/bin/tectonic -X compile main_revised.tex --outdir "$OUT" --keep-logs 2>&1 \
  | grep -viE "invalid utf-8|^note: downloading|special command|annotation|^warning: >>" | tail -4
[ -f "$OUT/main_revised.pdf" ] || { echo "no PDF -- see $OUT/main_revised.log"; exit 1; }
cp "$OUT/main_revised.pdf" "$PAPER/main_revised.pdf"
echo "PDF: $PAPER/main_revised.pdf"
grep -iE "Reference \`[^']*' .* undefined|Citation \`[^']*' .* undefined" "$OUT/main_revised.log" \
  | sed "s/LaTeX Warning: //" | sort -u | sed 's/^/undefined: /'
/home/seungju/miniforge3/envs/advTrain/bin/python - "$OUT/main_revised.pdf" <<'PY'
import re, sys
from pypdf import PdfReader
pages = [re.sub(r'\s+', ' ', p.extract_text() or '') for p in PdfReader(sys.argv[1]).pages]
app = next((i for i, t in enumerate(pages) if 'Proof and Scope of the Fixed-Anchor' in t), None)
ref = next((i for i, t in enumerate(pages) if len(re.findall(r'\b(?:19|20)\d\d[a-c]?\.', t)) >= 8), None)
print("total %d pages | appendix p%s | references p%s"
      % (len(pages), app + 1 if app is not None else '?', ref + 1 if ref is not None else '?'))
if app is not None:
    over = app - 9
    print("main body %d-%d pages against ICLR's 9 (10 with the extra page)%s"
          % (app, app + 1, "" if over <= 0 else "  --  %d-%d over" % (over, over + 1)))
PY
