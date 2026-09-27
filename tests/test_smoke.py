import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import dp_ppnn as P


def test_dataset_is_the_assessors_file():
    assert P.sha256(P.DATA_PATH) == P.CANONICAL_SHA256
    df = P.load_frame()
    assert df.shape == (4269, 13)
    assert int(df.loan_status.sum()) == 2656
    Xtr, Xte, ytr, yte, _, _ = P.split(df)
    assert Xtr.shape == (3415, 12) and Xte.shape == (854, 12)


def test_per_example_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    m = P.MLP(12, P.ARCHS["report"], rng)
    X, y = rng.normal(size=(1, 12)), np.array([1.0])
    (g,), _ = m.clipped_mean_grads(X, y, None, 1.0)
    def loss():
        z = m.logits(X)[0]
        return np.logaddexp(0, z) - y[0] * z
    W = m.W[1]; i, j = 3, 5; h = 1e-6
    W[i, j] += h; lp = loss(); W[i, j] -= 2 * h; lm = loss(); W[i, j] += h
    assert abs((lp - lm) / (2 * h) - g[2][i, j]) < 1e-7


def test_clipping_bounds_norm():
    rng = np.random.default_rng(1)
    m = P.MLP(12, P.ARCHS["report"], rng)
    X, y = rng.normal(size=(1, 12)) * 10, np.array([0.0])
    for clip in ["l1", "l2"]:
        (g,), _ = m.clipped_mean_grads(X, y, clip, 0.5)
        flat = np.concatenate([a.ravel() for a in g])
        nrm = np.abs(flat).sum() if clip == "l1" else np.linalg.norm(flat)
        assert nrm <= 0.5 + 1e-12


def test_calibration_and_ledger():
    lap = P.calibrate("laplace", "per_update", 2.0, 32, 10)
    assert np.isclose(lap.scale, 2 * 1.0 / (32 * 2.0)) and lap.ledger["per_record_total_eps"] == 20.0
    lit = P.calibrate("laplace", "literal", 0.2, 32, 10, clip="l2", n_params=2945)
    assert np.isclose(lit.scale, 5.0) and np.isclose(lit.ledger["per_update_eps_actual"], 2 * np.sqrt(2945) / 32 / 5.0)
    s = P.analytic_gaussian_sigma(1.0, 1e-5)
    assert 3.6 < s < 3.9                          # Balle & Wang (2018) analytic Gaussian
    tot_l = P.calibrate("laplace", "total", 8.0, 64, 5); tot_g = P.calibrate("gaussian", "total", 8.0, 64, 5)
    assert np.isclose(tot_l.scale, 5 * 2 / 64 / 8.0)
    assert np.isclose(tot_g.scale, np.sqrt(5) * P.analytic_gaussian_sigma(8.0, 1e-5) * 2 / 64)


def test_one_private_epoch_runs():
    df = P.load_frame(); Xtr, Xte, ytr, yte, _, _ = P.split(df); Xs, Xts = P.standardize(Xtr, Xte)
    plan = P.calibrate("laplace", "per_update", 8.0, 64, 1)
    m = P.train(Xs, ytr, batch=64, epochs=1, lr=0.01, seed=1, plan=plan, clip="l1", shards=2)
    r = P.metrics(yte, m.predict_proba(Xts))
    assert 0.0 <= r["roc_auc"] <= 1.0
