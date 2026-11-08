"""
seislab.py
==========
Python re-implementation of the MATLAB processing functions used in

    W. A. Mousa and A. A. Al-Shuhail, "Processing of Seismic Reflection Data
    Using MATLAB", Synthesis Lectures on Signal Processing, Morgan & Claypool, 2011.

The original executable implementation was unavailable for this revision.
These functions belong to the supplied Python reconstruction inspired by the
cited processing workflow. They are not a verified byte-for-byte translation
of the original MATLAB implementation.

Conventions
-----------
* D  : ndarray (nt, ntraces)  -- traces are columns, exactly like the book.
* hdr: dict of 1-D ndarrays   -- one entry per SEG-Y header word (cdp, offset, ...).
* dt : sampling interval in seconds.
"""
from __future__ import annotations

import numpy as np
import scipy.io as sio
from scipy.linalg import solve_toeplitz
from scipy.ndimage import uniform_filter1d
from scipy.signal import firwin, fftconvolve

# ---------------------------------------------------------------------------
# Chapter 2 - data loading / geometry
# ---------------------------------------------------------------------------
HEADER_WORDS = ("tracl", "fldr", "tracf", "cdp", "offset", "sx", "gx",
                "selev", "gelev", "scalco", "scalel", "dt", "ns")


def load_book_data(path: str):
    """Load Book_Seismic_Data.mat -> (D, hdr, dt, t)."""
    m = sio.loadmat(path)
    D = np.asarray(m["D"], dtype=np.float64)
    H = m["H"][0]
    hdr = {}
    for w in HEADER_WORDS:
        if w in H.dtype.names:
            hdr[w] = np.array([np.squeeze(h[w]).item() for h in H], dtype=np.float64)
    dt = hdr["dt"][0] * 1e-6                       # micro-seconds -> seconds
    t = np.arange(D.shape[0]) * dt
    return D, hdr, dt, t


def extracting_shots(D, hdr, shot_num):
    """Return the traces of one (or several) shot gathers."""
    shot_num = np.atleast_1d(shot_num)
    idx = np.where(np.isin(hdr["fldr"], shot_num))[0]
    return D[:, idx], hdr["offset"][idx], idx


# ---------------------------------------------------------------------------
# Chapter 3 - quality control: trace editing & gain
# ---------------------------------------------------------------------------
def trace_editing(D, mode="book"):
    """Mute (zero) bad traces.

    mode="book" reproduces the book listing literally:
        [i,j] = find(D == max(max(D)));  D(:,j) = 0;
    i.e. every trace that contains the global maximum sample is zeroed.
    (For the East-Texas line this removes the clipped near-offset traces.)
    mode="none" returns the data unchanged.
    Returns (D_edited, killed_trace_indices).
    """
    D = D.copy()
    if mode == "none":
        return D, np.array([], dtype=int)
    j = np.unique(np.where(D == D.max())[1])
    D[:, j] = 0.0
    return D, j


def iac(D, t, pow=2.0, T=0):
    """Data-independent amplitude correction (book Eq. 3.1 / 3.2).

    T=0 : f_c(t) = f(t) * t**pow      (pow=2 -> the t^2 spherical-divergence gain)
    T=1 : f_c(t) = f(t) * exp(pow*t)
    """
    g = t ** pow if T == 0 else np.exp(pow * t)
    return D * g[:, None]


def agc_gain(D, dt, agc_gate=0.5, T=2):
    """Automatic gain control (book Section 3.3).

    T=1 : RMS AGC          -- divide by the RMS inside a sliding gate
    T=2 : instantaneous AGC -- divide by the mean |amplitude| inside a sliding gate
    """
    L = max(int(round(agc_gate / dt)), 1)
    eps = np.finfo(float).tiny
    if T == 1:
        e = np.sqrt(uniform_filter1d(D ** 2, size=L, axis=0, mode="nearest"))
    else:
        e = uniform_filter1d(np.abs(D), size=L, axis=0, mode="nearest")
    out = np.zeros_like(D)
    ok = e > eps * 1e10
    out[ok] = D[ok] / e[ok]
    return out


# ---------------------------------------------------------------------------
# Chapter 4 - noise attenuation: FIR band-pass filter
# ---------------------------------------------------------------------------
def bpf_fir(D, dt, N=100, cut_off=(15.0, 60.0)):
    """FIR band-pass filter (book: N=100, cut_off=[15,60] Hz).

    Designed like MATLAB fir1(N, cut_off/(fs/2)) (Hamming window, N+1 taps).
    Applied with zero-phase alignment (the N/2 group delay is removed) so the
    reflection times are not shifted.
    """
    fs = 1.0 / dt
    b = firwin(N + 1, cut_off, pass_zero=False, fs=fs, window="hamming")
    full = fftconvolve(D, b[:, None], mode="full", axes=0)
    d = N // 2
    return full[d:d + D.shape[0], :]


# ---------------------------------------------------------------------------
# Chapter 5 - spiking deconvolution (Wiener, Eq. 5.12-5.16)
# ---------------------------------------------------------------------------
def spiking_decon(D, max_lag=0.2, mu=0.1, dt=0.002):
    """Trace-by-trace Wiener spiking deconvolution.

    max_lag : operator length in seconds (book: 0.2 s)
    mu      : percent pre-whitening added to the zero-lag autocorrelation (book: 0.1)
    The whole trace is used as autocorrelation window (as stated in the book).
    """
    nt, nx = D.shape
    L = int(round(max_lag / dt)) + 1
    out = np.zeros_like(D)
    nfft = int(2 ** np.ceil(np.log2(2 * nt)))
    F = np.fft.rfft(D, nfft, axis=0)
    R = np.fft.irfft(np.abs(F) ** 2, nfft, axis=0)[:L, :]   # autocorrelation lags 0..L-1
    for k in range(nx):
        r = R[:, k].copy()
        if r[0] <= 0:
            continue
        r[0] *= 1.0 + mu / 100.0
        g = np.zeros(L)
        g[0] = 1.0
        h = solve_toeplitz(r, g)
        y = np.convolve(D[:, k], h)[:nt]
        # keep the trace energy (purely cosmetic, AGC follows anyway)
        s = np.linalg.norm(y)
        out[:, k] = y * (np.linalg.norm(D[:, k]) / s if s > 0 else 0.0)
    return out


# ---------------------------------------------------------------------------
# Chapter 6 - CMP sorting, velocity analysis, NMO, stacking
# ---------------------------------------------------------------------------
def ssort(D, hdr):
    """Sort traces into CMP gathers (by cdp, then |offset|)."""
    order = np.lexsort((np.abs(hdr["offset"]), hdr["cdp"]))
    return D[:, order], {k: v[order] for k, v in hdr.items()}, order


def extracting_cmp(Dsort, hsort, cmp_num):
    idx = np.where(hsort["cdp"] == cmp_num)[0]
    return Dsort[:, idx], hsort["offset"][idx], idx


def extracting_cmp_fold_num(hsort):
    cmps, fold = np.unique(hsort["cdp"], return_counts=True)
    return cmps.astype(int), fold


def _moveout_gather(G, t, offset, v_of_t0):
    """Map every trace of gather G to t0 using t(x)=sqrt(t0^2 + x^2/v^2)."""
    nt, nx = G.shape
    out = np.empty_like(G)
    tx = np.sqrt(t[:, None] ** 2 + (offset[None, :] / v_of_t0[:, None]) ** 2)
    for j in range(nx):
        out[:, j] = np.interp(tx[:, j], t, G[:, j], left=0.0, right=0.0)
    return out, tx


def semblance(G, t, offset, velocities, win=0.04):
    """Velocity spectrum (normalised semblance, book Section 6.3)."""
    nt = len(t)
    dt = t[1] - t[0]
    L = max(int(round(win / dt)), 1)
    S = np.zeros((nt, len(velocities)))
    M = max(np.count_nonzero(np.any(G != 0, axis=0)), 1)
    for iv, v in enumerate(velocities):
        A, _ = _moveout_gather(G, t, offset, np.full(nt, v))
        es = uniform_filter1d(A.sum(1) ** 2, L, mode="nearest")
        eu = uniform_filter1d((A ** 2).sum(1), L, mode="nearest")
        S[:, iv] = np.where(eu > 0, es / (M * eu + 1e-300), 0.0)
    return S


def vel_picking(Dsort, hsort, t, vmin=5000.0, dv=200.0, nv=51,
                cmp_start=205, cmp_end=255, cmp_step=5, n_pts=8,
                t_min=0.25, smooth_pen=0.01, supergather=2):
    """Automatic replacement for the book's interactive (mouse) vel_picking.m.

    For every analysis CMP the semblance panel is computed with the book
    parameters (vmin=5000 ft/s, dv=200, nv=51) and a smooth maximum-semblance
    velocity path through it is found with dynamic programming. The path is
    then sampled at n_pts times, which mimics the n_pts=8 mouse picks.
    Returns (v_stack, t_stack, cmps) with v_stack/t_stack of shape (n_cmp, n_pts).
    """
    vels = vmin + dv * np.arange(nv)
    cmps = np.arange(cmp_start, cmp_end + 1, cmp_step)
    t_picks = np.linspace(t_min, t[-1] - 0.1, n_pts)
    v_stack = np.zeros((len(cmps), n_pts))
    for ic, c in enumerate(cmps):
        # 5-CMP super-gather for a more stable velocity spectrum
        idx = np.where(np.abs(hsort["cdp"] - c) <= supergather)[0]
        G, off = Dsort[:, idx], hsort["offset"][idx]
        S = semblance(G, t, off, vels, win=0.08)
        S[t < t_min] = 0.0
        S = S / (S.max(axis=1, keepdims=True) + 1e-12) * S.max(axis=1, keepdims=True) ** 0.5
        # dynamic programming over time (only every 5th sample for speed);
        # velocity is not allowed to decrease with time (monotonic V_rms)
        step = 5
        Ss = S[::step]
        ns, nvv = Ss.shape
        cost = -Ss.copy()
        back = np.zeros((ns, nvv), dtype=int)
        dif = np.arange(nvv)[:, None] - np.arange(nvv)[None, :]       # to - from
        pen = smooth_pen * dif.astype(float) ** 2
        pen[dif < 0] = np.inf
        for k in range(1, ns):
            tot = cost[k - 1][None, :] + pen          # (to, from)
            back[k] = np.argmin(tot, axis=1)
            cost[k] += tot[np.arange(nvv), back[k]]
        path = np.zeros(ns, dtype=int)
        path[-1] = np.argmin(cost[-1])
        for k in range(ns - 1, 0, -1):
            path[k - 1] = back[k, path[k]]
        vpath = vels[path]
        v_stack[ic] = np.interp(t_picks, t[::step], vpath)
    t_stack = np.tile(t_picks, (len(cmps), 1))
    return v_stack, t_stack, cmps


def velocity_field(hsort_cmps, t, v_stack, t_stack, cmps):
    """Interpolate picks in time and CMP -> v(t0) for every CMP (nearest outside)."""
    vt = np.array([np.interp(t, t_stack[i], v_stack[i]) for i in range(len(cmps))])
    field = {}
    for c in np.unique(hsort_cmps):
        cc = np.clip(c, cmps[0], cmps[-1])
        field[int(c)] = np.array([np.interp(cc, cmps, vt[:, k]) for k in range(len(t))])
    return field


def nmo_correction(Dsort, hsort, t, vfield, max_stretch=10.0):
    """NMO-correct every CMP gather and mute samples whose stretch exceeds
    max_stretch percent (book: max_stretch = 10)."""
    out = np.zeros_like(Dsort)
    for c, v in vfield.items():
        idx = np.where(hsort["cdp"] == c)[0]
        if idx.size == 0:
            continue
        G = Dsort[:, idx]
        off = hsort["offset"][idx]
        A, tx = _moveout_gather(G, t, off, v)
        with np.errstate(divide="ignore", invalid="ignore"):
            stretch = 100.0 * (tx - t[:, None]) / np.maximum(t[:, None], 1e-12)
        A[stretch > max_stretch] = 0.0
        out[:, idx] = A
    return out


def sstack(Dnmo, hsort):
    """Stack each CMP gather (average of live samples) -> (Dstacked, cmp_num)."""
    cmps = np.unique(hsort["cdp"]).astype(int)
    S = np.zeros((Dnmo.shape[0], len(cmps)))
    for k, c in enumerate(cmps):
        G = Dnmo[:, hsort["cdp"] == c]
        live = np.count_nonzero(G, axis=1)
        S[:, k] = np.where(live > 0, G.sum(1) / np.maximum(live, 1), 0.0)
    return S, cmps


# ---------------------------------------------------------------------------
# Full processing flow (book Section 2.4, steps 1-5,7,8)
# ---------------------------------------------------------------------------
class SDP:
    """The standard seismic data processing flow of the book.

    raw --> trace editing --> t^2 gain --> BPF 15-60 Hz --> spiking decon
        --> instantaneous AGC 0.5 s --> CMP sort --> NMO + stretch mute --> stack

    Velocity analysis is run once on the *clean* data (fit_velocities) and the
    same velocity field is then used for every distorted version, so that the
    stacked sections differ only because of the distortion.
    """

    def __init__(self, hdr, dt, t, edit_mode="book", gain_pow=2.0,
                 bpf_order=100, bpf_band=(15.0, 60.0), decon_lag=0.2,
                 decon_mu=0.1, agc_gate=0.5, max_stretch=10.0, kill=None):
        self.hdr, self.dt, self.t = hdr, dt, t
        self.edit_mode = edit_mode
        self.gain_pow, self.bpf_order, self.bpf_band = gain_pow, bpf_order, bpf_band
        self.decon_lag, self.decon_mu = decon_lag, decon_mu
        self.agc_gate, self.max_stretch = agc_gate, max_stretch
        self.kill = kill          # trace indices killed on the clean data
        self.vfield = None

    def prestack(self, D):
        if self.kill is None:
            D, self.kill = trace_editing(D, self.edit_mode)
        else:
            D = D.copy()
            D[:, self.kill] = 0.0
        D = iac(D, self.t, self.gain_pow, 0)
        D = bpf_fir(D, self.dt, self.bpf_order, self.bpf_band)
        D = spiking_decon(D, self.decon_lag, self.decon_mu, self.dt)
        D = agc_gain(D, self.dt, self.agc_gate, T=2)
        D[:, self.kill] = 0.0
        Ds, hs, _ = ssort(D, self.hdr)
        return Ds, hs

    def fit_velocities(self, D_clean, **kw):
        Ds, hs = self.prestack(D_clean)
        self.v_stack, self.t_stack, self.cmps = vel_picking(Ds, hs, self.t, **kw)
        self.vfield = velocity_field(hs["cdp"], self.t, self.v_stack, self.t_stack, self.cmps)
        return self

    def run(self, D):
        Ds, hs = self.prestack(D)
        Dn = nmo_correction(Ds, hs, self.t, self.vfield, self.max_stretch)
        S, cmps = sstack(Dn, hs)
        return S, cmps
