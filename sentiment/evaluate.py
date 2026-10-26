"""Step 6: read the test set once (or the out-of-domain set with --split waimai) and compare against the zero-shot baseline.

    uv run python baseline.py --split test --temperature <baseline T from validation>   # first
    uv run python evaluate.py --run runs/main

Our model runs with both option orders; metrics use the canonical order (negative first, as in calibration), and the
other order measures the flip rate. Writes runs/main/eval_test.json (everything) and results_rows.json (the table).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from common import DATA_DIR, LABELS, RUNS_DIR
from metrics import flip_rate, metrics, paired_bootstrap, softmax
from model import load_tokenizer, load_trained
from train import predict, read_rows

DOMAINS = {"hotel": "酒店|房间|入住|宾馆|前台|早餐|客房|携程", "book": "书|作者|小说|出版|读|故事",
           "laptop": "笔记本|电脑|屏|键盘|硬盘|驱动|系统|内存|电池|散热|机器"}


def domain(text):
    hits = [d for d, pat in DOMAINS.items() if re.search(pat, text)]
    return hits[0] if len(hits) == 1 else ("mixed" if hits else "unclear")


def reliability(p, labels, bins=10):
    conf, correct = p.max(1), p.argmax(1) == labels
    rows = []
    for lo in np.linspace(0.5, 0.95, bins):                    # two options: confidence is always >= 0.5
        m = (conf >= lo) & (conf < lo + 0.05 + (1e-9 if lo >= 0.95 else 0))
        if m.any():
            rows.append({"bin": f"{lo:.2f}-{lo + 0.05:.2f}", "n": int(m.sum()), "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", help="test, or waimai for the out-of-domain set")
    a = ap.parse_args()
    run = Path(a.run)
    test = read_rows(a.split)
    labels = np.array([LABELS.index(r["label"]) for r in test])
    tok = load_tokenizer()
    model, ckpt = load_trained(run)
    T = float(ckpt["temperature"])
    model.head.temperature = 1.0                              # raw logits; T is applied in metrics
    z = predict(model, tok, test, tok.pad_token_id or 0, order=LABELS)
    z_swap = predict(model, tok, test, tok.pad_token_id or 0, order=LABELS[::-1])

    base = json.loads((Path(RUNS_DIR) / "baseline" / f"{a.split}_logits.json").read_text(encoding="utf-8"))
    base_report = json.loads((Path(RUNS_DIR) / "baseline" / f"{a.split}.json").read_text(encoding="utf-8"))
    assert base["ids"] == [r["id"] for r in test], "baseline and test rows differ"
    zb = (np.array(base["pos_first"]) + np.array(base["neg_first"])) / 2
    Tb = base_report["order_avg"]["calibrated"]["temperature"]

    ours_raw, ours_cal = metrics(z, labels, 1.0, LABELS), metrics(z, labels, T, LABELS)
    base_cal = metrics(zb, labels, Tb, LABELS)
    p = softmax(z, T)
    correct = p.argmax(1) == labels
    correct_b = softmax(zb, Tb).argmax(1) == labels
    doms = [domain(r["text"]) for r in test]
    per_domain = {d: {"n": int(sum(x == d for x in doms)), "accuracy": float(np.mean([c for c, x in zip(correct, doms) if x == d])),
                      "baseline_accuracy": float(np.mean([c for c, x in zip(correct_b, doms) if x == d]))}
                  for d in sorted(set(doms))}
    wrong = np.nonzero(~correct)[0]
    worst = sorted(wrong, key=lambda i: -p[i].max())[:20]
    report = {
        "n": len(test), "temperature": T, "baseline_temperature": Tb,
        "ours_raw": ours_raw, "ours_calibrated": ours_cal, "baseline_calibrated": base_cal,
        "ours_flip_rate": flip_rate(z, z_swap), "baseline_flip_rate": base_report["flip_rate"],
        "accuracy_vs_baseline": paired_bootstrap(correct, correct_b),
        "per_domain": per_domain, "reliability": reliability(p, labels),
        "confident_errors": [{"id": test[i]["id"], "label": test[i]["label"], "p_label": float(p[i, labels[i]]),
                              "domain": doms[i], "text": test[i]["text"][:200]} for i in worst],
    }
    np.savez(run / f"logits_{a.split}.npz", ours=z, ours_swap=z_swap, baseline=zb, labels=labels)
    (run / f"eval_{a.split}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pick = lambda m, flip, name: {"model": name, "accuracy": m["accuracy"], "nll": m["nll"], "ece": m["ece"], "brier": m["brier"],
                                  "coverage_5": m["coverage_5"], "order_flip": flip}
    pre = "" if a.split == "test" else "外卖 OOD · "
    rows = [pick(base_cal, base_report["flip_rate"], f"{pre}Zero-shot baseline（T={Tb:.2f}）"),
            pick(ours_raw, report["ours_flip_rate"], f"{pre}Ours（T=1）"),
            pick(ours_cal, report["ours_flip_rate"], f"{pre}Ours（calibrated，T={T:.2f}）")]
    (run / ("results_rows.json" if a.split == "test" else f"results_rows_{a.split}.json")).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "confident_errors"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
