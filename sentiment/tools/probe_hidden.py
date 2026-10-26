"""How large are the backbone's hidden states at the readout positions, and what logits does a freshly initialized
pointer head produce on them? Explains the initial training loss.

    uv run python tools/probe_hidden.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR, LABELS  # noqa: E402
from model import POINTER_DIM, PointerHead, collate, encode, load_backbone, load_tokenizer  # noqa: E402

rows = [json.loads(l) for l in open(Path(DATA_DIR) / "train.jsonl", encoding="utf-8")][:16]
tok = load_tokenizer()
backbone = load_backbone().cuda().eval()
ids, mask, ends, decide = collate([encode(tok, r["text"]) for r in rows], tok.pad_token_id or 0, "cuda")
with torch.no_grad():
    h = backbone(input_ids=ids, attention_mask=mask).last_hidden_state.float()
b = torch.arange(len(rows), device="cuda")
hd, ho = h[b, decide], h[b[:, None], ends]
print(f"hidden size {h.shape[-1]} | |h_decide| mean {hd.norm(dim=-1).mean():.1f} | |h_opt| mean {ho.norm(dim=-1).mean():.1f} | "
      f"max |value| {h[mask.bool()].abs().max():.1f}")
torch.manual_seed(0)
for name, head in [("default init", PointerHead(h.shape[-1], POINTER_DIM).cuda()), ("zero-init W_k", PointerHead(h.shape[-1], POINTER_DIM, zero_init=True).cuda())]:
    head.train()
    z = head(hd, ho)
    y = torch.tensor([LABELS.index(r["label"]) for r in rows], device="cuda")
    print(f"{name:14s} | logit gap mean {(z[:, 0] - z[:, 1]).abs().mean():.2f} | initial CE {torch.nn.functional.cross_entropy(z, y):.3f}")
