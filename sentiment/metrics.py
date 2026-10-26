"""Metrics shared by the baseline, calibration and evaluation scripts. Everything works on raw logits [N, K] and integer
labels [N]; probabilities are softmax(logits / T)."""
import numpy as np

T_GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 121))     # the temperatures calibration tries


def softmax(logits, T=1.0):
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, labels, T=1.0):
    p = softmax(logits, T)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def fit_temperature(logits, labels):
    """The grid temperature with the lowest mean NLL. Scaling all logits by 1/T never changes the argmax."""
    losses = [nll(logits, labels, t) for t in T_GRID]
    return float(T_GRID[int(np.argmin(losses))])


def ece(conf, correct, bins=10):
    """Expected calibration error: per confidence bin, |accuracy - mean confidence|, weighted by the bin's share."""
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & ((conf < hi) if i < bins - 1 else (conf <= hi))
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def coverage_at_error(conf, correct, budget):
    """The largest share of items, taken most confident first, whose error rate stays within `budget`."""
    order = np.argsort(-np.asarray(conf), kind="stable")
    errors = np.cumsum(~np.asarray(correct)[order])
    ok = errors / np.arange(1, len(order) + 1) <= budget
    return float(np.nonzero(ok)[0].max() + 1) / len(order) if ok.any() else 0.0


def metrics(logits, labels, T=1.0, label_names=None):
    labels = np.asarray(labels)
    p = softmax(logits, T)
    pred, conf = p.argmax(1), p.max(1)
    correct = pred == labels
    onehot = np.eye(p.shape[1])[labels]
    out = {
        "n": int(len(labels)),
        "temperature": float(T),
        "accuracy": float(correct.mean()),
        "nll": nll(logits, labels, T),
        "brier": float(((p - onehot) ** 2).sum(1).mean()),
        "ece": ece(conf, correct),
        "confident_error": float(((conf >= 0.9) & ~correct).mean()),
        "coverage_5": coverage_at_error(conf, correct, 0.05),
        "coverage_1": coverage_at_error(conf, correct, 0.01),
        "mean_confidence": float(conf.mean()),
    }
    if label_names:
        for k, name in enumerate(label_names):
            m = labels == k
            out[f"recall_{name}"] = float(correct[m].mean()) if m.any() else None
    return out


def flip_rate(logits_a, logits_b):
    """How often two runs of the same items (e.g. the two option orders) disagree on the answer."""
    return float((np.asarray(logits_a).argmax(1) != np.asarray(logits_b).argmax(1)).mean())


def paired_bootstrap(correct_a, correct_b, samples=1000, seed=0):
    """Accuracy difference a - b with a 95% interval from resampling the same items for both."""
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(samples, len(a)))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    return {"diff": float(a.mean() - b.mean()), "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5))}
