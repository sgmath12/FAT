#!/usr/bin/env bash
# Compile writting_docs/paper locally and report the page budget (2026-09-02).
#
# tectonic is a single ~14 MB binary in ~/.local/bin -- not a TeX Live installation.  It downloads
# only the packages this document actually needs, caches them, and after the first run compiles in
# about twenty seconds.  The point is that page count can be checked here after every edit instead of
# by uploading to Overleaf to find out.
set -u
cd "$(dirname "$0")/../writting_docs/paper"
OUT="${1:-/tmp/fat_paper_build}"
mkdir -p "$OUT"
~/.local/bin/tectonic -X compile main.tex --outdir "$OUT" --keep-logs 2>&1 \
  | grep -viE "invalid utf-8|^note: downloading|special command|annotation|^warning: >>" | tail -6
/home/seungju/miniforge3/envs/advTrain/bin/python - "$OUT/main.pdf" <<'PY'
import sys, re
from pypdf import PdfReader
r = PdfReader(sys.argv[1]); pages = [p.extract_text() for p in r.pages]
app = next((i for i, t in enumerate(pages) if 'ALGORITHM' in re.sub(r'\s', '', t)[:400]), None)
ref = next((i for i, t in enumerate(pages) if 'REFERENCES' in re.sub(r'\s', '', t)), None)
print("total %d pages | appendix starts p%s | references p%s"
      % (len(r.pages), app + 1 if app is not None else '?', ref + 1 if ref is not None else '?'))
if app is not None:
    print("main body ~%d pages against ICLR's 9-10" % app)
PY
