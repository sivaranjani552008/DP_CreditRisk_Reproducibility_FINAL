#!/usr/bin/env bash
# One-command entry points.
#   ./run_all.sh verify   checksum, unit tests, number-by-number check of the response (seconds)
#   ./run_all.sh quick    smoke run of every study arm into a temporary folder (< 1 min)
#   ./run_all.sh full     full settings study + validation suite + summaries + response rebuild
set -euo pipefail
cd "$(dirname "$0")"
mode="${1:-verify}"
case "$mode" in
  verify)
    python scripts/verify_dataset.py
    python -m pytest -q tests
    python response/verify_response.py ;;
  quick)
    out="$(mktemp -d)"
    python experiments/run_study.py --quick --out "$out"
    python experiments/summarize.py --runs "$out" > /dev/null
    echo "quick run written to $out" ;;
  full)
    python scripts/verify_dataset.py
    python experiments/run_study.py --out results/study
    python experiments/gradient_geometry.py results/study
    python experiments/summarize.py --runs results/study
    python validation/validate_claims.py --out results/validated/validation
    python response/make_numbers.py
    node response/build_response.js
    python response/verify_response.py ;;
  *) echo "usage: $0 {verify|quick|full}"; exit 2 ;;
esac
