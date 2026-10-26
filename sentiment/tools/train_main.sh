#!/usr/bin/env bash
# The full training run: the plan's settings, with the head lr and W_k init chosen at R2.
source /mnt/g/dev/repo/xev/sentiment/env.sh
o=runs/main
rm -rf "$o"; mkdir -p "$o"
uv run python train.py --out "$o" > "$o/stdout.log" 2>&1
echo "exit=$?"
tail -3 "$o/stdout.log"
