"""Step 7: serve the trained classifier.

    uv run uvicorn serve:app --host 127.0.0.1 --port 8010         # XEV_RUN=runs/main by default

    curl -s localhost:8010/v1/sentiment -H 'content-type: application/json' -d '{"text": "房间很干净，下次还会再来。"}'
"""
import os
import time

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from common import LABELS
from model import collate, encode, load_tokenizer, load_trained

RUN = os.environ.get("XEV_RUN", "runs/main")
tok = load_tokenizer()
model, ckpt = load_trained(RUN)                 # LoRA merged in, temperature from calibration
app = FastAPI(title="xev-sentiment")


class Request(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


@app.get("/health")
def health():
    return {"run": RUN, "temperature": model.head.temperature, "step": ckpt.get("step")}


@app.post("/v1/sentiment")
@torch.no_grad()
def sentiment(req: Request):
    t0 = time.perf_counter()
    ids, mask, ends, decide = collate([encode(tok, req.text, LABELS)], tok.pad_token_id or 0, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = torch.softmax(model(ids, mask, ends, decide).float(), dim=-1)[0].tolist()
    k = len(LABELS)
    top = max(range(k), key=p.__getitem__)
    return {"label": LABELS[top], "probabilities": {name: round(v, 4) for name, v in zip(LABELS, p)},
            "confidence": round((p[top] - 1 / k) / (1 - 1 / k), 4), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
