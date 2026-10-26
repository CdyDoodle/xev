#!/usr/bin/env bash
# Out-of-domain: prepare waimai, run the baseline with its ChnSentiCorp-validation T, evaluate ours with its own T.
# No temperature or setting is refit on waimai: the point is how both models transfer as they are.
set -e
source /mnt/g/dev/repo/xev/sentiment/env.sh
uv run python prepare_ood.py > data/waimai_prepare.log 2>&1
echo "== prepared"; grep -E 'rows read|"kept"' data/waimai_prepare.log
TB=$(uv run python -c "import json; print(json.load(open('runs/baseline/validation.json'))['order_avg']['calibrated']['temperature'])")
uv run python baseline.py --split waimai --temperature "$TB" > runs/baseline/waimai.log 2>&1
echo "== baseline done"
uv run python evaluate.py --run runs/main --split waimai > runs/main/evaluate_waimai.log 2>&1
echo "== evaluate done"
