"""Distortion models: DCT lossy compression and AWGN at a target raw-data SNR."""
import numpy as np
from scipy.fft import dct, idct


def awgn(D, snr_db, rng):
    """Additive white Gaussian noise, SNR defined over the whole data set:
    sigma^2 = (sum D^2 / numel) / 10^(SNR/10)."""
    p = np.mean(D ** 2)
    sigma = np.sqrt(p / 10 ** (snr_db / 10.0))
    return D + sigma * rng.standard_normal(D.shape)


def _compress_columns(X, snr_db):
    """Orthonormal DCT-II of every column of X; in each column the lowest-energy
    coefficients are zeroed until the discarded energy reaches
    ||x||^2 / 10^(SNR/10) (largest number of coefficients that does not exceed
    it). Returns (X_rec, n_kept)."""
    C = dct(X, type=2, norm="ortho", axis=0)
    E = C ** 2
    order = np.argsort(E, axis=0)                          # ascending, per column
    Es = np.take_along_axis(E, order, axis=0)
    target = E.sum(0) / 10 ** (snr_db / 10.0)
    k = (np.cumsum(Es, axis=0) <= target[None, :]).sum(0)  # coefs dropped per column
    drop = np.arange(X.shape[0])[:, None] < k[None, :]
    mask = np.ones_like(C, dtype=bool)
    np.put_along_axis(mask, order, ~drop, axis=0)
    Cq = np.where(mask, C, 0.0)
    return idct(Cq, type=2, norm="ortho", axis=0), int(mask.sum())


def dct_compress(D, snr_db, mode="block", block=128):
    """Transform-based lossy compression of the raw data.

    mode='block'  (default): every trace is cut into non-overlapping time blocks
                  of `block` samples (JPEG-like block DCT); each block is DCT'd
                  and its smallest coefficients are discarded so that the block
                  meets the target SNR. This is a time-local block distortion model. The block size
                  is a reconstruction choice, not a verified original setting.
    mode='trace' : one DCT per whole trace, each trace meets the target SNR.
    mode='global': one DCT per trace, a single threshold for the whole data set.
    Returns (D_rec, compression_ratio = total coefficients / kept coefficients).
    """
    nt, nx = D.shape
    if mode == "global":
        C = dct(D, type=2, norm="ortho", axis=0)
        e = (C ** 2).ravel()
        target = e.sum() / 10 ** (snr_db / 10.0)
        order = np.argsort(e)
        k = int(np.searchsorted(np.cumsum(e[order]), target, side="right"))
        Cq = C.ravel().copy()
        Cq[order[:k]] = 0.0
        rec = idct(Cq.reshape(C.shape), type=2, norm="ortho", axis=0)
        return rec, e.size / max(e.size - k, 1)

    B = nt if mode == "trace" else block
    nb = int(np.ceil(nt / B))
    pad = np.zeros((nb * B, nx))
    pad[:nt] = D
    X = pad.reshape(nb, B, nx).transpose(1, 0, 2).reshape(B, nb * nx)   # blocks as columns
    live = np.any(X != 0, axis=0)
    Y = np.zeros_like(X)
    Y[:, live], kept = _compress_columns(X[:, live], snr_db)
    rec = Y.reshape(B, nb, nx).transpose(1, 0, 2).reshape(nb * B, nx)[:nt]
    return rec, X[:, live].size / max(kept, 1)
