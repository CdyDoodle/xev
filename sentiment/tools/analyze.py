"""Extra analysis for the report: accuracy by review length, how often both models confidently contradict the label
(a rough label-noise estimate), and whether the predicted positive rate matches the true one (a prior shift).

    uv run python tools/analyze.py          # needs runs/main/logits_{test,waimai}.npz from evaluate.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR, LABELS  # noqa: E402
from metrics import softmax  # noqa: E402
from model import load_tokenizer  # noqa: E402

BUCKETS = [(0, 10), (11, 30), (31, 100), (101, 10**9)]
TEMPS = json.loads(Path("runs/main/eval_test.json").read_text(encoding="utf-8"))


def main():
    tok = load_tokenizer()
    out = {}
    for split in ["test", "waimai"]:
        rows = [json.loads(l) for l in open(Path(DATA_DIR) / f"{split}.jsonl", encoding="utf-8")]
        d = np.load(f"runs/main/logits_{split}.npz")
        y = d["labels"]
        p_ours, p_base = softmax(d["ours"], TEMPS["temperature"]), softmax(d["baseline"], TEMPS["baseline_temperature"])
        c_ours, c_base = p_ours.argmax(1) == y, p_base.argmax(1) == y
        lens = np.array([len(t) for t in tok([r["text"] for r in rows], add_special_tokens=False).input_ids])
        by_len = []
        for lo, hi in BUCKETS:
            m = (lens >= lo) & (lens <= hi)
            if m.any():
                by_len.append({"tokens": f"{lo}-{hi}" if hi < 10**9 else f">{lo - 1}", "n": int(m.sum()),
                               "ours": float(c_ours[m].mean()), "baseline": float(c_base[m].mean())})
        both_contra = (p_ours[np.arange(len(y)), y] < 0.1) & (p_base[np.arange(len(y)), y] < 0.1)
        pos = LABELS.index("positive")
        out[split] = {
            "n": int(len(y)),
            "by_length": by_len,
            "both_confidently_contradict_label": int(both_contra.sum()),
            "both_contradict_share": float(both_contra.mean()),
            "accuracy_excluding_those": {"ours": float(c_ours[~both_contra].mean()), "baseline": float(c_base[~both_contra].mean())},
            "true_positive_rate": float((y == pos).mean()),
            "predicted_positive_rate": {"ours": float((p_ours.argmax(1) == pos).mean()), "baseline": float((p_base.argmax(1) == pos).mean())},
            "models_agree": float((p_ours.argmax(1) == p_base.argmax(1)).mean()),
            "only_ours_right": int((c_ours & ~c_base).sum()), "only_baseline_right": int((~c_ours & c_base).sum()),
            "examples_both_contradict": [rows[i]["text"][:80] + " ⟶ 标注 " + rows[i]["label"] for i in np.nonzero(both_contra)[0][:8]],
        }
    Path("runs/main/analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
