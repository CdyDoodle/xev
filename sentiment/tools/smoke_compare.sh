#!/usr/bin/env bash
# Two short smoke runs with a zero-initialized W_k, differing only in the pointer head's learning rate.
source /mnt/g/dev/repo/xev/sentiment/env.sh
for hl in 1e-3 1e-4; do
  o="runs/smoke-head$hl"
  rm -rf "$o"; mkdir -p "$o"
  uv run python train.py --out "$o" --limit 320 --max_steps 20 --warmup 0.25 --eval_every 20 --log_every 5 \
      --val_limit 100 --head_lr "$hl" > "$o/stdout.log" 2>&1
  echo "== head_lr $hl exit=$?"
  grep -E '^\{' "$o/stdout.log" | cut -c1-200
done
