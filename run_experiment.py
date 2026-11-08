"""
run_experiment.py
=================
Reconstruction of the experiments in
  "Novel Weighted Metric for Full-Reference Seismic Data Quality Assessment"
  (H. H. Nuha et al.)

For each data set (real East-Texas line, synthetic line with the same geometry)
and each distortion (DCT lossy compression, AWGN) at raw-data SNR = 50, 45, ..., 5 dB:

  1. distort the raw data                      x  -> x_hat
  2. full-reference metrics on raw data     NMSE(x_hat, x), eNMSE(x_hat, x)
  3. run the standard SDP flow on both         stack(x), stack(x_hat)
  4. "actual quality" on the stacked data      NMSE(stack(x_hat), stack(x))
  5. Pearson rho between each metric column and the actual-quality column

Usage
-----
    python run_experiment.py --data Book_Seismic_Data.mat --out results
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seislab import load_book_data, SDP
from metrics import nmse, enmse, pearson
from distortions import awgn, dct_compress
from synthetic import make_synthetic

SNRS = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]


def run_dataset(name, D, hdr, dt, t, args, rng):
    print(f"\n=== {name} data: {D.shape[1]} traces x {D.shape[0]} samples ===")
    edit = "book" if (name == "real" and args.edit == "book") else "none"
    sdp = SDP(hdr, dt, t, edit_mode=edit, max_stretch=args.max_stretch)
    sdp.fit_velocities(D)
    print(f"traces killed by trace editing: {len(sdp.kill)}")
    S_clean, cmps = sdp.run(D)

    rows, stacks, raws = [], {}, {}
    for snr in SNRS:
        row = {"SNR_dB": snr}
        for dist in ("DCT", "AWGN"):
            if dist == "DCT":
                Dd, cr = dct_compress(D, snr, args.dct_mode, args.dct_block)
                row["DCT_CR"] = cr
            else:
                Dd = awgn(D, snr, rng)
            S_d, _ = sdp.run(Dd)
            row[f"{dist}_NMSE"] = nmse(Dd, D)
            row[f"{dist}_eNMSE"] = enmse(Dd, D, dt)
            row[f"{dist}_Actual"] = nmse(S_d, S_clean)
            stacks[(dist, snr)] = S_d
            raws[(dist, snr)] = Dd
        rows.append(row)
        print(f"SNR {snr:2d} dB | DCT  NMSE={row['DCT_NMSE']:.3e} eNMSE={row['DCT_eNMSE']:.3e} "
              f"act={row['DCT_Actual']:.3e} | AWGN NMSE={row['AWGN_NMSE']:.3e} "
              f"eNMSE={row['AWGN_eNMSE']:.3e} act={row['AWGN_Actual']:.3e}")
    df = pd.DataFrame(rows)

    corr = {}
    for dist in ("DCT", "AWGN"):
        corr[dist] = {
            "rho_NMSE": pearson(df[f"{dist}_NMSE"], df[f"{dist}_Actual"]),
            "rho_eNMSE": pearson(df[f"{dist}_eNMSE"], df[f"{dist}_Actual"]),
            # supplementary (not in the paper): rank and log-domain correlation
            "spearman_NMSE": spearmanr(df[f"{dist}_NMSE"], df[f"{dist}_Actual"])[0],
            "spearman_eNMSE": spearmanr(df[f"{dist}_eNMSE"], df[f"{dist}_Actual"])[0],
            "logrho_NMSE": pearson(np.log10(df[f"{dist}_NMSE"]), np.log10(df[f"{dist}_Actual"])),
            "logrho_eNMSE": pearson(np.log10(df[f"{dist}_eNMSE"]), np.log10(df[f"{dist}_Actual"])),
        }
    # supplementary: pooled DCT + AWGN (20 points) -- can a single metric rank
    # distortions of different type correctly? (the Fig. 3 argument of the paper)
    act = np.r_[df["DCT_Actual"], df["AWGN_Actual"]]
    corr["pooled"] = {}
    for m in ("NMSE", "eNMSE"):
        x = np.r_[df[f"DCT_{m}"], df[f"AWGN_{m}"]]
        corr["pooled"][f"rho_{m}"] = pearson(x, act)
        corr["pooled"][f"spearman_{m}"] = spearmanr(x, act)[0]
        corr["pooled"][f"logrho_{m}"] = pearson(np.log10(x), np.log10(act))
    print("Correlation with actual quality:", json.dumps(corr, indent=1))
    return df, corr, S_clean, stacks, cmps, sdp


# ---------------------------------------------------------------------------
def fmt(v):
    """LaTeX number format similar to the manuscript."""
    if v == 0:
        return "0"
    if abs(v) >= 1e-3:
        return f"{v:.4f}"
    m, e = f"{v:.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def latex_table(df, caption, label):
    lines = [r"\begin{table*}[!t]", r"  \begin{center}", f"    \\caption{{{caption}}}",
             f"    \\label{{{label}}}", r"    \small", r"    \setlength{\tabcolsep}{4pt}",
             r"    \renewcommand{\arraystretch}{1.15}",
             r"    \begin{tabular}{|c|c|c|c|c|c|c|}", r"    \hline",
             r"    \multirow{2}{*}{\shortstack{Raw Data\\SNR (dB)}}",
             r"    & \multicolumn{3}{c|}{DCT}", r"    & \multicolumn{3}{c|}{AWGN} \\",
             r"    \cline{2-7}",
             r"    & NMSE & eNMSE & \shortstack{Actual\\Quality}",
             r"    & NMSE & eNMSE & \shortstack{Actual\\Quality} \\", r"    \hline \hline"]
    for _, r in df.iterrows():
        vals = [r["DCT_NMSE"], r["DCT_eNMSE"], r["DCT_Actual"],
                r["AWGN_NMSE"], r["AWGN_eNMSE"], r["AWGN_Actual"]]
        lines.append(f"    {int(r['SNR_dB'])} & " + " & ".join(fmt(v) for v in vals) + r" \\")
        lines.append(r"    \hline")
    lines += [r"    \end{tabular}", r"  \end{center}", r"\end{table*}"]
    return "\n".join(lines)


def plot_stacks(S_clean, stacks, cmps, t, snr, path, title):
    fig, ax = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    clip = np.percentile(np.abs(S_clean), 98)
    for a, (S, ttl) in zip(ax, [(S_clean, "Without distortion"),
                                (stacks[("DCT", snr)], f"DCT distortion ({snr} dB)"),
                                (stacks[("AWGN", snr)], f"AWGN distortion ({snr} dB)")]):
        a.imshow(S, aspect="auto", cmap="gray", vmin=-clip, vmax=clip,
                 extent=[cmps[0], cmps[-1], t[-1], 0])
        a.set_title(ttl)
        a.set_xlabel("CMP number")
    ax[0].set_ylabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_metrics(df, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, dist in zip(ax, ("DCT", "AWGN")):
        a.semilogy(df["SNR_dB"], df[f"{dist}_NMSE"], "o-", label="NMSE (raw)")
        a.semilogy(df["SNR_dB"], df[f"{dist}_eNMSE"], "s-", label="eNMSE (raw)")
        a.semilogy(df["SNR_dB"], df[f"{dist}_Actual"], "^-", label="Actual (stacked)")
        a.set_xlabel("Raw-data SNR (dB)")
        a.set_title(dist)
        a.invert_xaxis()
        a.grid(True, which="both", alpha=0.3)
        a.legend()
    ax[0].set_ylabel("Distortion")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_velocity(sdp, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, c in enumerate(sdp.cmps):
        ax.plot(sdp.v_stack[i], sdp.t_stack[i], ".-", label=f"CMP {c}")
    ax.invert_yaxis()
    ax.set_xlabel("Stacking velocity (ft/s)")
    ax.set_ylabel("t0 (s)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Book_Seismic_Data.mat")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--edit", default="book", choices=["book", "none"],
                    help="trace editing on real data: 'book' = literal book code")
    ap.add_argument("--max-stretch", type=float, default=10.0,
                    help="NMO stretch mute in percent (book listing: 10)")
    ap.add_argument("--fig-snr", type=int, default=40)
    ap.add_argument("--dct-mode", default="block", choices=["block", "trace", "global"],
                    help="DCT compression scheme (see distortions.py)")
    ap.add_argument("--dct-block", type=int, default=128,
                    help="block length in samples for --dct-mode block")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    D, hdr, dt, t = load_book_data(args.data)
    Dsyn = make_synthetic(hdr, D.shape[0], dt, seed=args.seed)

    summary = {}
    for name, data, cap, lab in [
            ("real", D, "Result for Different Type of Distortions (East Texas real data)",
             "tab:distortion_results"),
            ("synthetic", Dsyn, "Results for lossy compression distortion for synthetic data",
             "tab:lossy_synthetic")]:
        df, corr, S_clean, stacks, cmps, sdp = run_dataset(name, data, hdr, dt, t, args, rng)
        df.to_csv(os.path.join(args.out, f"table_{name}.csv"), index=False)
        with open(os.path.join(args.out, f"table_{name}.tex"), "w") as f:
            f.write(latex_table(df, cap, lab) + "\n")
        plot_stacks(S_clean, stacks, cmps, t, args.fig_snr,
                    os.path.join(args.out, f"stacked_{name}_{args.fig_snr}dB.png"),
                    f"Stacked sections, {name} data")
        plot_metrics(df, os.path.join(args.out, f"metrics_{name}.png"),
                     f"Raw-data metrics vs. actual quality, {name} data")
        plot_velocity(sdp, os.path.join(args.out, f"velocity_picks_{name}.png"))
        np.savez_compressed(os.path.join(args.out, f"sections_{name}_{args.fig_snr}dB.npz"),
                            reference=S_clean, DCT=stacks[("DCT", args.fig_snr)],
                            AWGN=stacks[("AWGN", args.fig_snr)], cmps=cmps, time_s=t)
        np.save(os.path.join(args.out, f"stack_clean_{name}.npy"), S_clean)
        summary[name] = corr
    np.save(os.path.join(args.out, "synthetic_raw.npy"), Dsyn)
    with open(os.path.join(args.out, "correlations.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
