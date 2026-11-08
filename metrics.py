"""Quality metrics used in the paper (Eq. 1-6)."""
import numpy as np


def mse(x_hat, x):
    """Eq. (1): MSE = 1/(MN) * sum_i ||x_hat_i - x_i||^2"""
    return np.mean((x_hat - x) ** 2)


def nmse(x_hat, x):
    """Eq. (2): NMSE = sum_i ||x_hat_i - x_i||^2 / sum_i ||x_i||^2"""
    return np.sum((x_hat - x) ** 2) / np.sum(x ** 2)


def snr_db(x_hat, x):
    """Eq. (3): SNR_dB = -10 log10(NMSE)"""
    return -10.0 * np.log10(nmse(x_hat, x))


def enmse_weights(nt, dt):
    """Weighting vector of the eNMSE: w(n) = (n*dt)^2, n = 0..N-1.

    This is the t^2 geometric-spreading gain of Mousa & Al-Shuhail (Eq. 3.1,
    alpha = 2), i.e. w = [0, dt^2, (2dt)^2, (3dt)^2, ..., ((N-1)dt)^2].
    NOTE: Eq. (5) of the manuscript prints [0, dt^2, 2dt^2, 3dt^2, ...]; the
    values in Table 1 (eNMSE/NMSE ~ 16 for AWGN = mean(t^4) over 0-3 s) are
    only reproduced with (n*dt)^2, so that is what is implemented here.
    """
    return (np.arange(nt) * dt) ** 2


def enmse(x_hat, x, dt, w=None):
    """Eq. (4): eNMSE = sum_i ||w o (x_hat_i - x_i)||^2 / sum_i ||x_i||^2
    (traces are the columns of x)."""
    if w is None:
        w = enmse_weights(x.shape[0], dt)
    return np.sum((w[:, None] * (x_hat - x)) ** 2) / np.sum(x ** 2)


def pearson(a, b):
    """Eq. (6): Pearson correlation coefficient."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])
