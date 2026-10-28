import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from utils import finish_time_model as ftm


def _frame(seed=0, n_courses=6):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_courses):
        km = rng.choice([42.2, 50.0, 80.5, 160.9])
        u = rng.normal(0, 0.15)
        for _ in range(rng.integers(3, 12)):
            g = rng.choice(["M", "F"])
            age = rng.integers(20, 70)
            pace = np.exp(2.0 + 0.1 * np.log(km / 50) + u + rng.normal(0, 0.2))
            rows.append({"name": f"c{c}", "gender": g, "age": age, "distance_km": km, "time_ms": pace * km * 60000})
    return pd.DataFrame(rows)


def _brute(frame, theta, sigma, tau):
    z, y = ftm.design(frame)
    out = {}
    for c, rows in frame.groupby("name").groups.items():
        i = frame.index.get_indexer(rows)
        g = frame["gender"].to_numpy()[i] == "F"
        mean = (z[i] * theta[g.astype(int)]).sum(1)
        cov = np.diag(sigma[g.astype(int)] ** 2) + tau**2
        out[c] = multivariate_normal(mean, cov).logpdf(y[i])
    return out


def test_collapsed_loglik_matches_brute_force():
    frame = _frame()
    stats = ftm.course_stats(frame)
    theta = np.array([[2.0, 0.12, 0.03, 0.01], [2.1, 0.09, 0.02, 0.015]])
    sigma, tau = np.array([0.2, 0.18]), 0.15
    ll = ftm.course_loglik(stats, theta[None], sigma[None], np.array([tau]))[0]
    brute = _brute(frame, theta, sigma, tau)
    np.testing.assert_allclose(ll, [brute[c] for c in stats.courses], rtol=1e-10)


def test_course_effect_posterior_matches_gaussian_conditioning():
    frame = _frame(seed=1, n_courses=3)
    stats = ftm.course_stats(frame)
    theta = np.array([[2.0, 0.1, 0.0, 0.0], [2.05, 0.1, 0.0, 0.0]])
    sigma, tau = np.array([0.2, 0.25]), 0.3
    mean, sd = ftm.course_effect_posterior(stats, theta[None], sigma[None], np.array([tau]))
    z, y = ftm.design(frame)
    for j, c in enumerate(stats.courses):
        i = np.flatnonzero(frame["name"] == c)
        g = (frame["gender"].to_numpy()[i] == "F").astype(int)
        r = y[i] - (z[i] * theta[g]).sum(1)
        prec = 1 / tau**2 + (1 / sigma[g] ** 2).sum()
        np.testing.assert_allclose(mean[0, j], (r / sigma[g] ** 2).sum() / prec)
        np.testing.assert_allclose(sd[0, j], prec**-0.5)


def test_features_subset_drops_age_terms():
    frame = _frame(seed=2)
    stats = ftm.course_stats(frame).features(2)
    theta = np.array([[2.0, 0.1], [2.1, 0.1]])
    full = np.column_stack([theta, np.zeros((2, 2))])
    a = ftm.course_loglik(stats, theta[None], np.array([[0.2, 0.2]]), np.array([0.1]))
    b = ftm.course_loglik(ftm.course_stats(frame), full[None], np.array([[0.2, 0.2]]), np.array([0.1]))
    np.testing.assert_allclose(a, b)
