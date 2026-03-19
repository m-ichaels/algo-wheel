#!/usr/bin/env bash
# Full pipeline: (data ->) signal -> calibration (phase A: wheel, pre-trade, gamma) -> handoff (phase B) -> tests -> figures -> summary -> report
# Usage: scripts/run_all.sh [--download] [--quick]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
Q=""; [[ " $* " == *" --quick "* ]] && Q="--quick"
[[ " $* " == *" --download "* ]] && python tools/download.py
python -u -m xcost signal | tee results/signal.log
python -u -m xcost calibrate $Q | tee results/calibrate.log
python -u -m xcost handoff $Q | tee results/handoff.log
python -m pytest -q | tee results/tests.txt
python scripts/plots.py
python scripts/summarize.py
python scripts/report.py
echo done
