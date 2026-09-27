#!/usr/bin/env python3
"""
dp_ppnn.py - paper-aligned, auditable implementation of the DP-PPNN framework of
Naresh, Reddi & Thamarai (2026), J. Supercomputing 82:516.

Pure NumPy (float64), exact per-example gradients, no deep-learning framework, so every
noise draw, clipping factor and privacy calibration is visible in ~400 lines.

Paper element                        -> implementation here
-----------------------------------------------------------------------------------------
Sect. 6 network (12 predictors, ReLU, sigmoid, BCE)      MLP(arch="report" 12-64-32-1 or
                                                           arch="keras_literal" 12-12-64-32-1)
Sect. 6 optimiser Adam, lr 0.001                          Adam (torch-identical update rule)
Alg. 1 step e (w <- w - eta * grad)                       optimizer="sgd"
Alg. 1 sequential minibatch training                      train(..., shards=1)
Alg. 2 disjoint partitions, per-partition noise, average  train(..., shards=Np)
Alg. 1/2 clipping ||g||_2 <= C (as printed)               clip="l2"
L1 clipping (needed for an L1-calibrated Laplace)         clip="l1"
Alg. 3 output perturbation                                output_perturbation()
Sect. 4.2 step 2 / step 4 input perturbation              input_perturbation()
Privacy calibration                                       calibrate() -> noise scale + ledger

Calibration modes (calibrate):
  "literal"    Laplace scale C/eps exactly as printed in Algorithms 1-2 (sensitivity
               S = C regardless of batch size or clipping norm).  The ledger reports the
               eps that this scale actually guarantees for the released minibatch mean.
  "per_update" eps is the privacy of ONE released update (the convention of Lemma 1 /
               Theorem 1).  Sensitivity of the mean of B clipped gradients under
               replace-one adjacency: 2C/B (in the clipping norm).  Laplace b = 2C/(B eps);
               Gaussian std from the analytic Gaussian mechanism (Balle & Wang 2018) at
               (eps, delta/K), K = number of epochs, so both mechanisms compose (basic
               composition) to the same per-record total (K*eps, delta) - Laplace with
               delta = 0.
  "total"      eps is the per-record budget for the WHOLE training run.  With disjoint
               shuffled minibatches each record is used K = epochs times (no subsampling
               amplification is claimed).  Laplace: pure-DP basic composition,
               b = K * 2C/(B eps)  (conservative: tighter Laplace accountants exist but are
               not used).  Gaussian: exact composition of K Gaussian releases,
               std = sqrt(K) * sigma_AG(eps, delta) * 2C/B (tight).  The accounting is
               therefore tilted in favour of the Gaussian mechanism.
"""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "loan_approval_dataset.csv"
CANONICAL_SHA256 = "4b5cd093d178378f4cfa8c107adb6e599b88be9d8a3b51f3b99c0d5914154e54"  # assessor's file
FEATURES_12 = ["loan_id", "no_of_dependents", "education", "self_employed", "income_annum",
               "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
               "commercial_assets_value", "luxury_assets_value", "bank_asset_value"]
# Public, data-independent bounds used only by the input-perturbation stage (features -> [0, 1]).
DOMAIN_BOUNDS = {"loan_id": (0.0, 5000.0), "no_of_dependents": (0.0, 5.0), "education": (0.0, 1.0),
                 "self_employed": (0.0, 1.0), "income_annum": (0.0, 10_000_000.0),
                 "loan_amount": (0.0, 40_000_000.0), "loan_term": (0.0, 25.0),
                 "cibil_score": (300.0, 900.0), "residential_assets_value": (-1_000_000.0, 30_000_000.0),
                 "commercial_assets_value": (0.0, 20_000_000.0),
                 "luxury_assets_value": (0.0, 40_000_000.0), "bank_asset_value": (0.0, 15_000_000.0)}
DELTA = 1e-5


# ============================================================================ data
def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frame(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])  # tolerate trailing commas
    if df.columns.tolist() != FEATURES_12 + ["loan_status"]:
        raise ValueError(f"unexpected columns {df.columns.tolist()}")
    for c in ["education", "self_employed", "loan_status"]:
        df[c] = df[c].astype(str).str.strip()
    df["education"] = df["education"].map({"Graduate": 1.0, "Not Graduate": 0.0})
    df["self_employed"] = df["self_employed"].map({"Yes": 1.0, "No": 0.0})
    df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})
    if df.isna().any().any():
        raise ValueError("unexpected missing / unmapped values")
    return df


def split(df: pd.DataFrame, seed: int = 42, features: Sequence[str] = FEATURES_12):
    y = df["loan_status"].to_numpy(np.int64)
    X = df[list(features)].to_numpy(np.float64)
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=seed, stratify=y)
    return X[tr], X[te], y[tr], y[te], tr, te


def standardize(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)  # training rows only
    return sc.transform(Xtr), sc.transform(Xte)


def to_unit_box(X, features: Sequence[str] = FEATURES_12):
    lo = np.array([DOMAIN_BOUNDS[f][0] for f in features])
    hi = np.array([DOMAIN_BOUNDS[f][1] for f in features])
    return np.clip((X - lo) / (hi - lo), 0.0, 1.0)


# ============================================================================ privacy calibration
@lru_cache(maxsize=None)
def analytic_gaussian_sigma(eps: float, delta: float) -> float:
    """Smallest sigma with N(0, sigma^2), L2-sensitivity 1, being (eps, delta)-DP
    (Balle & Wang 2018, Thm 8); bisection on the exact privacy profile."""
    def delta_of(s):
        a, b = 1.0 / (2 * s), eps * s
        return norm.cdf(a - b) - math.exp(eps) * norm.cdf(-a - b)
    lo, hi = 1e-6, 1e7
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if delta_of(mid) > delta else (lo, mid)
    return hi


def gaussian_eps(std_over_sens: float, delta: float) -> float:
    """eps of one Gaussian release with noise multiplier std/sensitivity at the given delta."""
    lo, hi = 1e-6, 1e4
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if analytic_gaussian_sigma(mid, delta) <= std_over_sens else (mid, hi)
    return hi


@dataclass
class NoisePlan:
    mechanism: str             # "laplace" | "gaussian" | "none"
    scale: float               # Laplace b, or Gaussian std, applied to each coordinate of the released mean
    ledger: Dict[str, float] = field(default_factory=dict)


def calibrate(mechanism: str, mode: str, eps: float, batch: int, epochs: int, clip_norm: float = 1.0,
              clip: str = "l1", delta: float = DELTA, n_params: Optional[int] = None,
              shards: int = 1, gaussian_multiplier: Optional[float] = None) -> NoisePlan:
    """Noise for the released (per-shard) mean of clipped per-example gradients."""
    if mechanism == "none":
        return NoisePlan("none", 0.0, {"per_record_total_eps": math.inf})
    m = max(1, batch // shards)                       # records averaged inside one released vector
    sens = 2.0 * clip_norm / m                        # replace-one sensitivity, in the clipping norm
    K = epochs
    L = {"mode": mode, "clip": clip, "eps_param": eps, "batch": batch, "shards": shards, "epochs": K,
         "sensitivity_in_clip_norm": sens}
    if mode == "literal":
        if mechanism == "laplace":                    # Algorithm 1/2 as printed: Laplace(C/eps)
            b = clip_norm / eps
            l1_sens = sens if clip == "l1" else sens * math.sqrt(n_params)  # ||v||_1 <= sqrt(d)||v||_2
            eps_step = l1_sens / b
            L.update(per_update_eps_actual=eps_step, per_update_delta=0.0,
                     per_record_total_eps=eps_step * K, per_record_total_delta=0.0)
            return NoisePlan("laplace", b, L)
        sigma = 1.1 if gaussian_multiplier is None else gaussian_multiplier   # DP-SGD setting of Sect. 6
        std = sigma * clip_norm / m
        eps_step = gaussian_eps(std / sens, delta / K)
        L.update(noise_multiplier=sigma, per_update_eps_actual=eps_step, per_update_delta=delta / K,
                 per_record_total_eps=eps_step * K, per_record_total_delta=delta)
        return NoisePlan("gaussian", std, L)
    if clip == "l2" and mechanism == "laplace":
        raise ValueError("an L1-calibrated Laplace release needs L1 clipping")
    if mode == "per_update":
        if mechanism == "laplace":
            b = sens / eps
            L.update(per_update_eps_actual=eps, per_update_delta=0.0,
                     per_record_total_eps=eps * K, per_record_total_delta=0.0)
            return NoisePlan("laplace", b, L)
        s = analytic_gaussian_sigma(eps, delta / K) * sens
        L.update(per_update_eps_actual=eps, per_update_delta=delta / K,
                 per_record_total_eps=eps * K, per_record_total_delta=delta)
        return NoisePlan("gaussian", s, L)
    if mode == "total":
        if mechanism == "laplace":
            b = K * sens / eps
            L.update(per_update_eps_actual=eps / K, per_update_delta=0.0,
                     per_record_total_eps=eps, per_record_total_delta=0.0, accountant="basic (pure DP)")
            return NoisePlan("laplace", b, L)
        s = math.sqrt(K) * analytic_gaussian_sigma(eps, delta) * sens
        L.update(per_record_total_eps=eps, per_record_total_delta=delta,
                 accountant="exact Gaussian composition (mu-GDP)")
        return NoisePlan("gaussian", s, L)
    raise ValueError(mode)


# ============================================================================ model
ARCHS = {"report": [64, 32], "keras_literal": [12, 64, 32]}


class MLP:
    """ReLU MLP with one sigmoid output; PyTorch-default initialisation U(-1/sqrt(fan_in), +)."""

    def __init__(self, n_in: int, hidden: Sequence[int], rng: np.random.Generator):
        dims = [n_in] + list(hidden) + [1]
        self.W, self.b = [], []
        for a, c in zip(dims[:-1], dims[1:]):
            bound = 1.0 / math.sqrt(a)
            self.W.append(rng.uniform(-bound, bound, size=(a, c)))
            self.b.append(rng.uniform(-bound, bound, size=(c,)))

    @property
    def params(self) -> List[np.ndarray]:
        return [p for pair in zip(self.W, self.b) for p in pair]

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def forward(self, X):
        acts, zs = [X], []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            zs.append(z)
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
            acts.append(h)
        return acts, zs

    def logits(self, X):
        return self.forward(X)[1][-1][:, 0]

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(self.logits(X), -500, 500)))

    def clipped_mean_grads(self, X, y, clip: Optional[str], C: float, groups: Optional[List[np.ndarray]] = None):
        """Exact per-example gradients of BCE-with-logits, clipped per example
        (factor min(1, C/||g_i||)), averaged within each group of row indices.
        Per-example norms use ||a (x) d||_2 = ||a||_2 ||d||_2 and ||a (x) d||_1 = ||a||_1 ||d||_1,
        so per-example gradients never need to be materialised."""
        acts, zs = self.forward(X)
        p = 1.0 / (1.0 + np.exp(-np.clip(zs[-1][:, 0], -500, 500)))
        delta = (p - y)[:, None]
        deltas = [None] * len(self.W)
        for l in range(len(self.W) - 1, -1, -1):
            deltas[l] = delta
            if l > 0:
                delta = (delta @ self.W[l].T) * (zs[l - 1] > 0)
        n = X.shape[0]
        if clip is None:
            factor = np.ones(n)
            norms = np.zeros(n)
        else:
            if clip == "l2":
                sq = sum(((acts[l] ** 2).sum(1) + 1.0) * (deltas[l] ** 2).sum(1) for l in range(len(self.W)))
                norms = np.sqrt(sq)
            elif clip == "l1":
                norms = sum((np.abs(acts[l]).sum(1) + 1.0) * np.abs(deltas[l]).sum(1) for l in range(len(self.W)))
            else:
                raise ValueError(clip)
            factor = np.minimum(1.0, C / np.maximum(norms, 1e-12))
        groups = groups or [np.arange(n)]
        out = []
        for g in groups:
            f = factor[g][:, None] / len(g)
            grads = []
            for l in range(len(self.W)):
                d = deltas[l][g] * f
                grads.append(acts[l][g].T @ d)
                grads.append(d.sum(0))
            out.append(grads)
        return out, norms


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


class SGD:
    def __init__(self, params, lr):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


def train(X, y, *, batch: int, epochs: int, lr: float, seed: int, plan: NoisePlan,
          clip: Optional[str] = "l1", C: float = 1.0, arch: str = "report", optimizer: str = "adam",
          shards: int = 1, record_norms: bool = False):
    """Algorithm 1 (shards=1) / Algorithm 2 (shards=Np): per-example clipping, per-shard mean,
    per-shard noise, average of shards, optimiser step.  Initialisation and minibatch order
    depend only on `seed`; the noise stream is separate, so runs that differ only in the
    mechanism are paired (common random numbers)."""
    ss = np.random.SeedSequence(seed)
    r_init, r_order, r_noise = [np.random.default_rng(s) for s in ss.spawn(3)]
    model = MLP(X.shape[1], ARCHS[arch], r_init)
    params = model.params
    opt = Adam(params, lr) if optimizer == "adam" else SGD(params, lr)
    n = len(X)
    norm_log = []
    for _ in range(epochs):
        perm = r_order.permutation(n)
        for st in range(0, n, batch):
            idx = perm[st:st + batch]
            groups = [np.asarray(g) for g in np.array_split(np.arange(len(idx)), min(shards, len(idx)))]
            shard_grads, norms = model.clipped_mean_grads(X[idx], y[idx], clip, C, groups)
            if record_norms:
                norm_log.append(norms)
            agg = [np.zeros_like(p) for p in params]
            for grads in shard_grads:
                for a, g in zip(agg, grads):
                    if plan.mechanism == "laplace":
                        g = g + r_noise.laplace(0.0, plan.scale, size=g.shape)
                    elif plan.mechanism == "gaussian":
                        g = g + r_noise.normal(0.0, plan.scale, size=g.shape)
                    a += g / len(shard_grads)
            opt.step(params, agg)
    return (model, norm_log) if record_norms else model


# ============================================================================ stages 1 and 3
def input_perturbation(X01: np.ndarray, mechanism: str, eps: float, rng: np.random.Generator,
                       convention: str = "per_feature", delta: float = DELTA) -> Tuple[np.ndarray, Dict]:
    """Sect. 4.2 step 2 / step 4: noise on records whose features lie in [0, 1].
    convention="per_feature": Eq. (2) of the article - sensitivity 1 and budget eps for each
        feature separately (record-level budget d*eps by basic composition for Laplace;
        the Gaussian release is (eps, delta) per feature).
    convention="record": one release of the whole d-vector with record-level budget eps:
        L1 sensitivity d (Laplace), L2 sensitivity sqrt(d) (Gaussian)."""
    d = X01.shape[1]
    if convention == "per_feature":
        s1, s2, led = 1.0, 1.0, {"feature_eps": eps, "record_eps_basic": d * eps}
    else:
        s1, s2, led = float(d), math.sqrt(d), {"record_eps": eps}
    if mechanism == "laplace":
        scale = s1 / eps
        noisy = X01 + rng.laplace(0.0, scale, size=X01.shape)
    else:
        scale = analytic_gaussian_sigma(eps, delta) * s2
        noisy = X01 + rng.normal(0.0, scale, size=X01.shape)
    led.update(mechanism=mechanism, noise_scale=scale,
               noise_std=scale * math.sqrt(2) if mechanism == "laplace" else scale)
    return noisy, led


def output_perturbation(prob: np.ndarray, mechanism: str, eps: float, rng: np.random.Generator,
                        delta: float = DELTA, sensitivity: float = 1.0):
    """Algorithm 3: Private Prediction = f(x) + noise, sensitivity S of a probability = 1."""
    if mechanism == "laplace":
        scale = sensitivity / eps
        return prob + rng.laplace(0.0, scale, size=prob.shape), scale
    scale = analytic_gaussian_sigma(eps, delta) * sensitivity
    return prob + rng.normal(0.0, scale, size=prob.shape), scale


# ============================================================================ metrics
def metrics(y, score) -> Dict[str, float]:
    p = np.clip(score, 1e-7, 1 - 1e-7)
    pred = (score >= 0.5).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, score)),
            "log_loss": float(log_loss(y, p, labels=[0, 1]))}
