"""Step 1: download ChnSentiCorp, clean it, check for leakage between splits, and write JSONL.

    uv run python prepare_data.py

Writes data/{train,validation,test}.jsonl, one {"id", "text", "label"} per line (label = "negative" | "positive"),
and data/stats.json with counts, label balance and length percentiles (characters and base-model tokens).
"""
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from common import BASE, BASE_REVISION, DATA_DIR, DATASET, DATASET_REVISION, LABELS


def normalize(text):
    """The key used to find duplicates: case-folded, whitespace-collapsed."""
    return " ".join(text.casefold().split())


def percentiles(values):
    a = np.asarray(values)
    return {"p50": int(np.percentile(a, 50)), "p90": int(np.percentile(a, 90)), "p99": int(np.percentile(a, 99)), "max": int(a.max())}


def main():
    out = Path(DATA_DIR)
    out.mkdir(exist_ok=True)
    ds = load_dataset(DATASET, revision=DATASET_REVISION)
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)

    # A review whose copies carry different labels anywhere in the corpus has no trustworthy label: drop every copy.
    labels_of = {}
    for split in ds:
        for ex in ds[split]:
            text = (ex["text"] or "").strip()
            if text:
                labels_of.setdefault(normalize(text), set()).add(ex["label"])
    conflicted = {k for k, v in labels_of.items() if len(v) > 1}

    stats, seen_train = {"conflicting_texts": len(conflicted)}, set()
    for split in ["train", "validation", "test"]:            # train first: the other splits are checked against it
        rows, seen, dropped = [], set(), {"empty": 0, "conflict": 0, "duplicate": 0, "in_train": 0}
        for i, ex in enumerate(ds[split]):
            text = (ex["text"] or "").strip()
            if not text:
                dropped["empty"] += 1; continue
            key = normalize(text)
            if key in conflicted:
                dropped["conflict"] += 1; continue
            if key in seen:
                dropped["duplicate"] += 1; continue
            if split != "train" and key in seen_train:
                dropped["in_train"] += 1; continue          # the same review in train would inflate the score
            seen.add(key)
            rows.append({"id": f"{split}-{i}", "text": text, "label": LABELS[ex["label"]]})
        if split == "train":
            seen_train = seen

        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        tokens = [len(t) for t in tok([r["text"] for r in rows], add_special_tokens=False).input_ids]
        stats[split] = {
            "kept": len(rows), "dropped": dropped,
            "labels": {name: sum(r["label"] == name for r in rows) for name in LABELS},
            "chars": percentiles([len(r["text"]) for r in rows]),
            "tokens": percentiles(tokens),
        }

    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    for name in LABELS:
        sample = [r for r in map(json.loads, open(out / "train.jsonl", encoding="utf-8")) if r["label"] == name][:3]
        print(f"\n--- {name} examples")
        for r in sample:
            print(" ", r["text"][:120])


if __name__ == "__main__":
    main()
