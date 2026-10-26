"""Out-of-domain set: Chinese food-delivery reviews (waimai_10k), cleaned the same way as ChnSentiCorp.

    uv run python prepare_ood.py

Reads the repo's original waimai_10k.csv at a pinned revision (the Hub viewer's parquet copy is mojibake), drops empty
texts, label conflicts and duplicates, and any review that also appears in our ChnSentiCorp train split.
Writes data/waimai.jsonl and data/waimai_stats.json. Nothing here is used for training or for picking any setting.
"""
import csv
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from common import BASE, BASE_REVISION, DATA_DIR, LABELS
from prepare_data import normalize, percentiles

OOD = "XiangPan/waimai_10k"
OOD_REVISION = "dbb125403842b8924d864f09f1c0eb357bd57435"


def main():
    path = hf_hub_download(OOD, "waimai_10k.csv", repo_type="dataset", revision=OOD_REVISION)
    with open(path, encoding="utf-8-sig", newline="") as f:
        raw = [(int(r["label"]), (r["review"] or "").strip()) for r in csv.DictReader(f)]
    cjk = sum(any("一" <= ch <= "鿿" for ch in t) for _, t in raw)
    print(f"{len(raw)} rows read; {cjk} contain Chinese characters (a decoding check)")

    labels_of = {}
    for y, t in raw:
        if t:
            labels_of.setdefault(normalize(t), set()).add(y)
    conflicted = {k for k, v in labels_of.items() if len(v) > 1}
    train_keys = {normalize(json.loads(l)["text"]) for l in open(Path(DATA_DIR) / "train.jsonl", encoding="utf-8")}

    rows, seen, dropped = [], set(), {"empty": 0, "conflict": 0, "duplicate": 0, "in_train": 0}
    for i, (y, t) in enumerate(raw):
        if not t:
            dropped["empty"] += 1; continue
        k = normalize(t)
        if k in conflicted:
            dropped["conflict"] += 1; continue
        if k in seen:
            dropped["duplicate"] += 1; continue
        if k in train_keys:
            dropped["in_train"] += 1; continue
        seen.add(k)
        rows.append({"id": f"waimai-{i}", "text": t, "label": LABELS[y]})
    with open(Path(DATA_DIR) / "waimai.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    tokens = [len(t) for t in tok([r["text"] for r in rows], add_special_tokens=False).input_ids]
    stats = {"waimai": {"kept": len(rows), "dropped": dropped, "labels": {n: sum(r["label"] == n for r in rows) for n in LABELS},
                        "chars": percentiles([len(r["text"]) for r in rows]), "tokens": percentiles(tokens)},
             "conflicting_texts": len(conflicted), "source": f"{OOD}@{OOD_REVISION}"}
    (Path(DATA_DIR) / "waimai_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
