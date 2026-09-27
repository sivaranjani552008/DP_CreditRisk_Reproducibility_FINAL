#!/usr/bin/env python3
"""Measures the L1/L2 norm ratio of per-example gradients of the 12-64-32-1 network during
non-private training (B = 32, lr 0.001, first 2 epochs, 5 seeds).  This ratio decides whether
L1-clipped Laplace or L2-clipped Gaussian noise gives the better signal-to-noise ratio."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import dp_ppnn as P

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "study"
df = P.load_frame(); Xtr, Xte, ytr, yte, _, _ = P.split(df); Xs, _ = P.standardize(Xtr, Xte)
ratios, l2s = [], []
for seed in [101, 202, 303, 404, 505]:
    kw = dict(batch=32, epochs=2, lr=0.001, seed=seed, plan=P.NoisePlan("none", 0.0), C=1e12, record_norms=True)
    _, n1 = P.train(Xs, ytr, clip="l1", **kw)
    _, n2 = P.train(Xs, ytr, clip="l2", **kw)   # same seed -> same trajectory (no clipping is active)
    n1, n2 = np.concatenate(n1), np.concatenate(n2)
    ratios.append(n1 / n2); l2s.append(n2)
r, l2 = np.concatenate(ratios), np.concatenate(l2s)
res = {"n_params": 2945, "sqrt_n_params": float(np.sqrt(2945)), "n_gradients": int(len(r)),
       "l1_over_l2_median": float(np.median(r)), "l1_over_l2_p10": float(np.percentile(r, 10)),
       "l1_over_l2_p90": float(np.percentile(r, 90)), "l2_norm_median": float(np.median(l2))}
json.dump(res, open(out / "gradient_geometry.json", "w"), indent=2)
print(res)
