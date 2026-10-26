"""Step 5: fit the temperature T on validation and write it into the run's head.pt.

    uv run python calibrate.py --run runs/main

T is the grid value (0.25..4, 121 log-spaced points) with the lowest validation NLL. Dividing both logits by T never
changes which option wins, so accuracy is untouched; only the confidence moves.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import LABELS
from metrics import fit_temperature, metrics
from model import load_tokenizer, load_trained
from train import predict, read_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    tok = load_tokenizer()
    model, ckpt = load_trained(run)
    model.head.temperature = 1.0
    val = read_rows("validation")
    labels = np.array([LABELS.index(r["label"]) for r in val])
    z = predict(model, tok, val, tok.pad_token_id or 0)
    T = fit_temperature(z, labels)
    report = {"temperature": T, "raw": metrics(z, labels, 1.0, LABELS), "calibrated": metrics(z, labels, T, LABELS)}
    ckpt["temperature"] = T
    torch.save(ckpt, run / "head.pt")
    (run / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(run / "validation_logits.npy", z)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
