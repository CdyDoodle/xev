#!/usr/bin/env bash
# Re-run evaluation (same frozen model, same data, deterministic) only to save per-item logits for the report.
set -e
source /mnt/g/dev/repo/xev/sentiment/env.sh
for s in test waimai; do
  uv run python evaluate.py --run runs/main --split "$s" > "runs/main/evaluate_$s.log" 2>&1
  echo "== $s done"
done
