"""Step 4: post-train Qwen3.5-4B-Base into a sentiment classifier (LoRA + pointer head), our own loop.

    uv run python train.py --out runs/smoke --limit 200 --max_steps 20 --eval_every 20 --val_limit 100   # smoke run
    uv run python train.py --out runs/main                                                                # full run

Every optimizer step: micro-batches of `--micro` reviews, each with a freshly drawn option order (so the head cannot
learn a position), cross-entropy over the two options, gradients accumulated `--accum` times, then clip, AdamW and the
warmup + cosine schedule. Every `--eval_every` steps the model is scored on validation and the checkpoint with the
lowest validation NLL is kept (LoRA adapter + head.pt).
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import BASE, BASE_REVISION, DATA_DIR, LABELS
from metrics import ece, nll
from model import (MAX_TEXT_TOKENS, POINTER_DIM, PointerHead, SentimentModel, add_lora, collate, encode,
                   load_backbone, load_tokenizer)


def read_rows(split, limit=None):
    rows = [json.loads(l) for l in open(Path(DATA_DIR) / f"{split}.jsonl", encoding="utf-8")]
    return rows[:limit] if limit else rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def predict(model, tok, rows, pad_id, batch=16, order=LABELS):
    """Logits [N, 2] in LABELS order, with the options presented in `order`."""
    model.eval()
    out = []
    for s in range(0, len(rows), batch):
        encs = [encode(tok, r["text"], order) for r in rows[s:s + batch]]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(*collate(encs, pad_id, "cuda")).float()
        z = z[:, [order.index(name) for name in LABELS]]          # back to (negative, positive)
        out.append(z.cpu().numpy())
    model.train()
    return np.concatenate(out)


def save_checkpoint(model, out, step, config, val):
    model.backbone.save_pretrained(out)                           # the LoRA adapter only
    torch.save({"head": model.head.state_dict(), "pointer_dim": POINTER_DIM, "temperature": 1.0,
                "base": BASE, "base_revision": BASE_REVISION, "step": step, "val": val, "config": config}, out / "head.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=1e-4, help="1e-3 made the loss spike to ~13 in the smoke runs (hidden-state norms are ~160)")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--p_swap", type=float, default=0.5, help="probability of presenting the options in the other order")
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training reviews (smoke run)")
    ap.add_argument("--val_limit", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zero_init_head", type=int, choices=[0, 1], default=1, help="start W_k at zero so the first guess is 50/50")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    train, val = read_rows("train", a.limit), read_rows("validation", a.val_limit)
    val_labels = np.array([LABELS.index(r["label"]) for r in val])
    steps_per_epoch = math.ceil(len(train) / (a.micro * a.accum))
    total = a.max_steps or steps_per_epoch * a.epochs
    warmup = max(1, int(total * a.warmup))
    config = {**vars(a), "train_rows": len(train), "val_rows": len(val), "total_steps": total, "warmup_steps": warmup,
              "max_text_tokens": MAX_TEXT_TOKENS, "train_sha256": sha256(Path(DATA_DIR) / "train.jsonl")}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    t0 = time.time()
    tok = load_tokenizer()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    backbone = load_backbone(torch.bfloat16)
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone = add_lora(backbone)
    head = PointerHead(backbone.config.hidden_size, POINTER_DIM, zero_init=a.zero_init_head)
    model = SentimentModel(backbone, head).cuda().train()
    lora_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in lora_params) + sum(p.numel() for p in head.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"loaded in {time.time() - t0:.0f}s | trainable {trainable:,} of {total_params:,} ({trainable / total_params:.2%}) | "
          f"LoRA dtype {lora_params[0].dtype} | steps {total} (warmup {warmup})", flush=True)

    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr}, {"params": list(head.parameters()), "lr": a.head_lr}],
                            weight_decay=a.weight_decay)
    def schedule(step):                                           # linear warmup, then cosine decay to 0
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    def write(row):
        log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True)

    best, step, rng = float("inf"), 0, random.Random(a.seed)
    run_loss, run_correct, run_n, run_tokens, t_log = 0.0, 0, 0, 0, time.time()
    torch.cuda.reset_peak_memory_stats()
    while step < total:
        order_idx = list(range(len(train))); rng.shuffle(order_idx)
        for s in range(0, len(order_idx), a.micro * a.accum):
            group = [train[i] for i in order_idx[s:s + a.micro * a.accum]]
            for m in range(0, len(group), a.micro):
                micro = group[m:m + a.micro]
                orders = [LABELS[::-1] if rng.random() < a.p_swap else LABELS for _ in micro]
                encs = [encode(tok, r["text"], o) for r, o in zip(micro, orders)]
                y = torch.tensor([o.index(r["label"]) for r, o in zip(micro, orders)], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model(*collate(encs, pad_id, "cuda")).float()
                loss = F.cross_entropy(z, y)
                (loss * len(micro) / len(group)).backward()          # mean over the whole optimizer step
                run_loss += loss.item() * len(micro); run_correct += int((z.argmax(1) == y).sum()); run_n += len(micro)
                run_tokens += sum(len(e["ids"]) for e in encs)
            torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), a.clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0 or step == total:
                dt = time.time() - t_log
                write({"step": step, "train_loss": run_loss / run_n, "train_acc": run_correct / run_n,
                       "lr": sched.get_last_lr()[0], "tok_s": run_tokens / dt, "mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                       "elapsed_min": (time.time() - t0) / 60})
                run_loss, run_correct, run_n, run_tokens, t_log = 0.0, 0, 0, 0, time.time()
            if step % a.eval_every == 0 or step == total:
                z_val = predict(model, tok, val, pad_id)
                v = {"step": step, "val_nll": nll(z_val, val_labels), "val_acc": float((z_val.argmax(1) == val_labels).mean()),
                     "val_ece": ece(np.max(np.exp(z_val - z_val.max(1, keepdims=True)) / np.exp(z_val - z_val.max(1, keepdims=True)).sum(1, keepdims=True), 1),
                                    z_val.argmax(1) == val_labels)}
                if v["val_nll"] < best:
                    best = v["val_nll"]; save_checkpoint(model, out, step, config, v); v["saved"] = True
                write(v)
            if step >= total:
                break
    (out / "summary.json").write_text(json.dumps({"best_val_nll": best, "steps": step, "minutes": (time.time() - t0) / 60,
                                                  "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}, indent=2), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
