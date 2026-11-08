"""Synthetic 2-D land line with the same acquisition geometry as the East-Texas data.

Model (all in ft, s):
* 9 gently dipping reflectors, RMS velocity increasing with depth,
  hyperbolic moveout t(x) = sqrt(t0^2 + x^2/v^2), Ricker wavelet 30 Hz,
* geometric-spreading + absorption amplitude decay  ~ exp(-a t) / t
  (so the raw synthetic shows the same amplitude decay as real data),
* a direct/refracted first arrival (v = 7000 ft/s),
* ground roll: low-velocity (2500 ft/s), low-frequency (8 Hz), high amplitude.
This is the reference before added DCT distortion or AWGN; it includes fixed
first arrivals and ground roll.
"""
import numpy as np


def ricker(f, dt, length=0.2):
    tt = np.arange(-length / 2, length / 2 + dt / 2, dt)
    a = (np.pi * f * tt) ** 2
    return (1 - 2 * a) * np.exp(-a)


def _add_event(D, j, t_arr, amp, wav, dt):
    nt = D.shape[0]
    half = len(wav) // 2
    i0 = t_arr / dt
    base = int(np.floor(i0))
    frac = i0 - base
    for k, w in enumerate(wav):
        n = base + k - half
        if 0 <= n < nt - 1:
            D[n, j] += amp * w * (1 - frac)
            D[n + 1, j] += amp * w * frac


def make_synthetic(hdr, nt, dt, seed=0):
    rng = np.random.default_rng(seed)
    ntr = len(hdr["offset"])
    D = np.zeros((nt, ntr))
    t = np.arange(nt) * dt
    w_ref = ricker(30.0, dt)
    w_gr = ricker(8.0, dt, 0.5)
    w_fb = ricker(20.0, dt)

    t0s = np.array([0.35, 0.60, 0.85, 1.10, 1.40, 1.70, 2.00, 2.35, 2.70])
    vrms = 5800 + 2600 * t0s                     # ft/s, increasing with time
    rc = rng.choice([-1, 1], len(t0s)) * rng.uniform(0.5, 1.0, len(t0s))
    dips = rng.uniform(-2e-5, 2e-5, len(t0s))    # s/ft (gentle dip in t0 along CMP)

    xm = 0.5 * (hdr["sx"] + hdr["gx"])           # midpoint (ft)
    xm0 = xm.mean()
    for j in range(ntr):
        x = abs(hdr["offset"][j])
        for t0, v, r, p in zip(t0s, vrms, rc, dips):
            t0j = t0 + p * (xm[j] - xm0)
            tx = np.sqrt(t0j ** 2 + (x / v) ** 2)
            amp = r * np.exp(-0.4 * tx) / max(tx, 0.05)
            _add_event(D, j, tx, amp, w_ref, dt)
        # first break (direct / refraction)
        _add_event(D, j, 0.02 + x / 7000.0, 4.0 / max(0.02 + x / 7000.0, 0.05), w_fb, dt)
        # ground roll
        tg = 0.05 + x / 2500.0
        _add_event(D, j, tg, 25.0 * np.exp(-x / 6000.0), w_gr, dt)
    return D
