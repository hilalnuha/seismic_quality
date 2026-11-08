"""Export full-width seismic sections from the saved reconstruction arrays."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def export_sections():
    root = Path(__file__).resolve().parent
    output = root / "publication_figures"
    output.mkdir(exist_ok=True)
    settings = {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
    for number, name in ((3, "real"), (4, "synthetic")):
        arrays = np.load(root / "results" / f"sections_{name}_40dB.npz")
        cmps = arrays["cmps"]
        times = arrays["time_s"]
        display_limit = float(np.percentile(np.abs(arrays["reference"]), 98))
        if display_limit <= 0:
            raise ValueError("The reference display amplitude must be positive.")

        with plt.rc_context(settings):
            figure, axes = plt.subplots(
                1, 3, figsize=(7.2, 3.55), sharex=True, sharey=True
            )
            panels = (
                ("reference", "(a) Reference"),
                ("DCT", "(b) DCT, 40 dB"),
                ("AWGN", "(c) AWGN, 40 dB"),
            )
            for axis, (key, title) in zip(axes, panels):
                # Share one reference-derived amplitude scale across all panels.
                artist = axis.imshow(
                    arrays[key] / display_limit,
                    aspect="auto",
                    cmap="gray",
                    vmin=-1,
                    vmax=1,
                    extent=[cmps[0] - 0.5, cmps[-1] + 0.5, times[-1], 0],
                    interpolation="nearest",
                )
                axis.set_title(title, pad=7)
                axis.set_xlabel("CMP number", labelpad=4)
                axis.set_xticks([205, 225, 245, 265])
                axis.set_yticks(np.arange(0, 3.01, 0.5))
                axis.tick_params(length=3, pad=3)
                for spine in axis.spines.values():
                    spine.set_linewidth(0.7)

            axes[0].set_ylabel("Time (s)", labelpad=4)
            figure.subplots_adjust(
                left=0.075, right=0.985, bottom=0.24, top=0.895, wspace=0.09
            )
            colorbar_axis = figure.add_axes([0.34, 0.10, 0.35, 0.025])
            colorbar = figure.colorbar(
                artist, cax=colorbar_axis, orientation="horizontal", ticks=[-1, 0, 1]
            )
            colorbar.set_label("Normalized display amplitude", labelpad=2, fontsize=8)
            colorbar.ax.tick_params(labelsize=8, length=2, pad=2)
            colorbar.outline.set_linewidth(0.5)
            for extension in ("png", "pdf", "svg"):
                figure.savefig(
                    output / f"Figure_{number}_{name}_stacked_sections.{extension}",
                    dpi=600,
                    facecolor="white",
                )
            plt.close(figure)


if __name__ == "__main__":
    export_sections()
