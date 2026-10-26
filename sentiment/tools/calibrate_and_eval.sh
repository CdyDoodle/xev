#!/usr/bin/env bash
# Step 5 then Step 6: fit T on validation, run the baseline on test with its validation T, then evaluate our model on test.
set -e
source /mnt/g/dev/repo/xev/sentiment/env.sh
uv run python calibrate.py --run runs/main > runs/main/calibrate.log 2>&1
echo "== calibrate done"; grep -E '"temperature"' runs/main/calibration.json | head -1
TB=$(uv run python -c "import json; print(json.load(open('runs/baseline/validation.json'))['order_avg']['calibrated']['temperature'])")
echo "== baseline T from validation: $TB"
uv run python baseline.py --split test --temperature "$TB" > runs/baseline/test.log 2>&1
echo "== baseline test done"
uv run python evaluate.py --run runs/main > runs/main/evaluate.log 2>&1
echo "== evaluate done"
