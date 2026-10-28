"""
Collapsed finish-time model (nb 1).

Log pace of finisher i (gender g, course c) is

    y_i = theta_g . z_i + u_c + e_i,    u_c ~ N(0, tau^2),    e_i ~ N(0, sigma_g^2)

with z_i = [1, log(distance_km / 50), A_i, A_i^2][:K] and A_i = (age_i - 40) / 10. Every course
has one distance, so the course effect u_c integrates out in closed form: per course, the
residuals r = y - z.theta are jointly normal with covariance diag(sigma_g^2) + tau^2 11'. The
likelihood then needs only per (course, gender) sums of z z', z y and y^2, so the model fits
every finisher (over a million rows) with a handful of parameters, and the course effects
are recovered exactly afterwards from their conditional posterior.

Pace is minutes per km. Only Gaussian noise on log pace collapses this way; heavier tails
would need the course effects sampled explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REFERENCE_KM = 50.0
AGE_CENTER, AGE_SCALE = 40.0, 10.0
GENDERS = ("M", "F")
FEATURES = ("intercept", "log_distance", "age", "age_sq")


@dataclass
class CourseStats:
    """Per (course, gender) sufficient statistics; arrays are (course, gender, ...)."""

    courses: np.ndarray       # course ids (the model frame's `name`)
    log_distance: np.ndarray  # (course,)
    zz: np.ndarray            # (C, 2, 4, 4) sum of z z'
    zy: np.ndarray            # (C, 2, 4) sum of z y
    yy: np.ndarray            # (C, 2) sum of y^2

    @property
    def n(self) -> np.ndarray:
        return self.zz[:, :, 0, 0]

    def features(self, k: int) -> "CourseStats":
        """The same statistics restricted to the first k features (2 = no age, 3 = linear age)."""
        return CourseStats(self.courses, self.log_distance, self.zz[:, :, :k, :k], self.zy[:, :, :k], self.yy)


def design(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """z (rows x 4) and y (log min/km) for finishers in a process_results frame."""
    x = np.log(frame["distance_km"].to_numpy() / REFERENCE_KM)
    a = (frame["age"].to_numpy() - AGE_CENTER) / AGE_SCALE
    z = np.column_stack([np.ones(len(frame)), x, a, a**2])
    y = np.log(frame["time_ms"].to_numpy() / 60000 / frame["distance_km"].to_numpy())
    return z, y


def course_stats(frame: pd.DataFrame) -> CourseStats:
    """Sufficient statistics from finishers (finished, with time_ms, age, gender M/F)."""
    z, y = design(frame)
    c_idx, courses = pd.factorize(frame["name"], sort=True)
    g_idx = frame["gender"].map({g: i for i, g in enumerate(GENDERS)}).to_numpy()
    cells = c_idx * 2 + g_idx
    n_cells = len(courses) * 2
    zz = np.zeros((n_cells, 4, 4))
    np.add.at(zz, cells, z[:, :, None] * z[:, None, :])
    zy = np.zeros((n_cells, 4))
    np.add.at(zy, cells, z * y[:, None])
    yy = np.bincount(cells, weights=y**2, minlength=n_cells)
    log_distance = pd.Series(z[:, 1]).groupby(c_idx).first().to_numpy()
    return CourseStats(np.asarray(courses), log_distance, zz.reshape(-1, 2, 4, 4), zy.reshape(-1, 2, 4), yy.reshape(-1, 2))


def _pieces(stats: CourseStats, theta, sigma, tau, xp):
    """Per-course log-likelihood and the course-effect posterior pieces, for one parameter
    set (xp = numpy or pytensor.tensor). theta (2, K), sigma (2,), tau scalar."""
    zz, zy, yy, n = stats.zz, stats.zy, stats.yy, stats.n
    th = theta[None, :, :]                                            # (1, 2, K)
    rr = yy - 2 * (zy * th).sum(-1) + ((zz * th[:, :, None, :]).sum(-1) * th).sum(-1)  # sum r^2
    r1 = zy[:, :, 0] - (zz[:, :, 0, :] * th).sum(-1)                  # sum r
    w = 1.0 / sigma**2
    s = (w[None, :] * n).sum(-1)                                      # (C,)
    t = (w[None, :] * r1).sum(-1)
    within = (-0.5 * w[None, :] * rr - n * xp.log(sigma)[None, :]).sum(-1)
    marginal = -0.5 * xp.log1p(tau**2 * s) + 0.5 * tau**2 * t**2 / (1 + tau**2 * s)
    loglik = within + marginal - 0.5 * n.sum(-1) * np.log(2 * np.pi)
    return loglik, s, t


def course_loglik_pt(stats: CourseStats, theta, sigma, tau):
    """Per-course marginal log-likelihood as a pytensor expression (for pm.Potential)."""
    import pytensor.tensor as pt

    return _pieces(stats, theta, sigma, tau, pt)[0]


def course_loglik(stats: CourseStats, theta: np.ndarray, sigma: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Per-course marginal log-likelihood for draws: theta (S, 2, K), sigma (S, 2), tau (S,)
    -> (S, C). Used for LOO (leave one course out) and power-scaling."""
    return np.stack([_pieces(stats, theta[i], sigma[i], tau[i], np)[0] for i in range(len(tau))])


def course_effect_posterior(stats: CourseStats, theta: np.ndarray, sigma: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Conditional posterior of the course effects for each draw: mean and sd, each (S, C)."""
    means, sds = [], []
    for i in range(len(tau)):
        _, s, t = _pieces(stats, theta[i], sigma[i], tau[i], np)
        prec = 1 + tau[i] ** 2 * s
        means.append(tau[i] ** 2 * t / prec)
        sds.append(tau[i] / np.sqrt(prec))
    return np.stack(means), np.stack(sds)


def within_course_ss(stats: CourseStats, theta: np.ndarray) -> np.ndarray:
    """Within (course, gender) sum of squared residuals around the cell mean, for one theta
    (2, K): under the model it is sigma_g^2 * chi^2(n - 1). Shape (C, 2)."""
    th = theta[None, :, :]
    rr = stats.yy - 2 * (stats.zy * th).sum(-1) + ((stats.zz * th[:, :, None, :]).sum(-1) * th).sum(-1)
    r1 = stats.zy[:, :, 0] - (stats.zz[:, :, 0, :] * th).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return rr - np.where(stats.n > 0, r1**2 / np.maximum(stats.n, 1), 0.0)
