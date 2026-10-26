#!/usr/bin/env bash
source /mnt/g/dev/repo/xev/sentiment/env.sh
uv run python tools/analyze.py 2>&1 | grep -v -i warning
