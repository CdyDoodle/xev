#!/usr/bin/env bash
# Show what the pinned base model snapshot holds in the Hugging Face cache.
source /mnt/g/dev/repo/xev/sentiment/env.sh
d="$HF_HOME/hub/models--Qwen--Qwen3.5-4B-Base"
ls -la "$d"/snapshots/*/
du -shL "$d"/snapshots/*
