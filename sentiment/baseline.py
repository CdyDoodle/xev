"""Step 2: the zero-shot baseline. The untrained base model keeps its LM head and answers a multiple-choice prompt;
we read the next-token logits of the letters "A" and "B" (the classic way to get a probability out of an LLM).

    uv run python baseline.py --split validation
    uv run python baseline.py --split test --temperature <T fitted on validation>     # Step 6 only

Both option orders are run (positive as A, then negative as A) to measure the base model's order bias.
Writes runs/baseline/<split>.json (metrics) and <split>_logits.json (per-item logits for both orders).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from common import BASE, BASE_REVISION, DATA_DIR, LABELS, RUNS_DIR
from metrics import fit_temperature, flip_rate, metrics
from model import MAX_TEXT_TOKENS, load_tokenizer, user_ids

NAMES = {"negative": "负面", "positive": "正面"}
ORDERS = {"pos_first": ["positive", "negative"], "neg_first": ["negative", "positive"]}


def prompt_ids(tok, text, order):
    body = user_ids(tok, text)
    if len(body) > MAX_TEXT_TOKENS:
        body = body[: MAX_TEXT_TOKENS // 2] + body[-(MAX_TEXT_TOKENS - MAX_TEXT_TOKENS // 2):]
    tail = f"\n这条评论的情感是？\nA. {NAMES[order[0]]}\nB. {NAMES[order[1]]}\n答案："
    return tok("评论：", add_special_tokens=False).input_ids + body + tok(tail, add_special_tokens=False).input_ids


def letter_ids(tok, letter):
    """Every single-token way the model might write the letter next ("A", " A")."""
    ids = set()
    for s in (letter, " " + letter):
        enc = tok(s, add_special_tokens=False).input_ids
        if len(enc) == 1:
            ids.add(enc[0])
    return sorted(ids)


@torch.no_grad()
def run(lm, tok, rows, order, batch=16):
    """-> logits [N, 2] in LABELS order (negative, positive) for one option order."""
    letters = [letter_ids(tok, "A"), letter_ids(tok, "B")]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out = []
    for s in range(0, len(rows), batch):
        encs = [prompt_ids(tok, r["text"], order) for r in rows[s:s + batch]]
        L = max(map(len, encs))
        ids = torch.full((len(encs), L), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, e in enumerate(encs):                           # right padding: pads come after the real tokens
            ids[i, : len(e)] = torch.tensor(e); mask[i, : len(e)] = 1
        ids, mask = ids.cuda(), mask.cuda()
        h = lm.model(input_ids=ids, attention_mask=mask).last_hidden_state
        last = h[torch.arange(len(encs), device=h.device), mask.sum(1) - 1]        # the position right after "答案："
        logits = lm.lm_head(last).float()
        z = torch.stack([torch.logsumexp(logits[:, l], dim=1) for l in letters], dim=1)   # [B, 2]: A, B
        out.append(z.cpu().numpy())
    z = np.concatenate(out)
    by_label = np.zeros_like(z)
    for pos, name in enumerate(order):                         # map letter slots back to label order
        by_label[:, LABELS.index(name)] = z[:, pos]
    return by_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--temperature", type=float, default=None, help="apply this T (fitted on validation); default: fit on this split")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(Path(DATA_DIR) / f"{a.split}.jsonl", encoding="utf-8")]
    labels = np.array([LABELS.index(r["label"]) for r in rows])
    tok = load_tokenizer()
    t0 = time.time()
    lm = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    print(f"loaded in {time.time() - t0:.0f}s, letters A={letter_ids(tok, 'A')} B={letter_ids(tok, 'B')}", flush=True)

    logits = {name: run(lm, tok, rows, order) for name, order in ORDERS.items()}
    avg = (logits["pos_first"] + logits["neg_first"]) / 2     # averaging over both orders removes the order bias
    report = {"split": a.split, "n": len(rows), "flip_rate": flip_rate(logits["pos_first"], logits["neg_first"])}
    for name, z in {**logits, "order_avg": avg}.items():
        T = a.temperature if a.temperature is not None else fit_temperature(z, labels)
        report[name] = {"raw": metrics(z, labels, 1.0, LABELS), "calibrated": metrics(z, labels, T, LABELS)}
        report[name]["calibrated"]["temperature_source"] = "given" if a.temperature is not None else f"fit on {a.split}"
    out = Path(RUNS_DIR) / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.split}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"{a.split}_logits.json").write_text(json.dumps({"ids": [r["id"] for r in rows], "labels": labels.tolist(),
                                                           **{k: v.tolist() for k, v in logits.items()}}), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
