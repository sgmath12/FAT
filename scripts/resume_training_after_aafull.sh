#!/usr/bin/env bash
# Resumes the suspended abl_ce_freezehead run once the full-set AA evaluation finishes.
set -u
while kill -0 1687069 2>/dev/null; do sleep 30; done
kill -CONT 1686680 2>/dev/null && echo "=== $(date '+%m-%d %H:%M') training resumed ==="
