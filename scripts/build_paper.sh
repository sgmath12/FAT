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
r = PdfReader(sys.argv[1])
# The ICLR style prints line numbers down the margin, so a page's extracted text STARTS with a run of
# integers and any "first N characters" test finds nothing.  Match on content unique to each section
# instead, anywhere on the page.
pages = [re.sub(r'\s+', ' ', p.extract_text() or '') for p in r.pages]


def first(*needles):
    return next((i for i, t in enumerate(pages) if any(n in t for n in needles)), None)


app = first('Input: dataset')   # Algorithm 1, the first thing after \\appendix
# The REFERENCES heading is set in small caps and does not survive text extraction, so the
# bibliography is found by its shape instead: a page carrying many "..., 2021." year terminators.
ref = next((i for i, t in enumerate(pages)
            if len(re.findall(r'\b(?:19|20)\d\d[a-c]?\.', t)) >= 8), None)
print("total %d pages | appendix p%s | references p%s"
      % (len(pages), app + 1 if app is not None else '?', ref + 1 if ref is not None else '?'))
# hyperref's bookmarks give an exact section map, which the extracted text cannot -- the headings are
# set in small caps and do not survive extraction.
try:
    def walk(o, d=0):
        for it in o:
            if isinstance(it, list):
                walk(it, d + 1)
            elif d == 0:
                print("    p%-3d %s" % (r.get_destination_page_number(it) + 1, it.title))
    print("sections:")
    walk(r.outline)
except Exception:
    pass
if app is not None:
    # A float can straddle the boundary -- Algorithm 1 sits on the same page as the end of the
    # conclusion -- so the body is reported as the range it can be, not a single number.
    over = app - 9
    print("main body %d-%d pages against ICLR's 9 (10 with the extra page)%s"
          % (app, app + 1, "" if over <= 0 else "  --  %d-%d over" % (over, over + 1)))
PY
