import argparse
from pathlib import Path
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PlottingStyle"))
from SNOplus_PythonPublicationStyle import SNOplus_style
from twomodels_comparison import extract_residuals


def load_residuals(filepath):
    data = np.load(filepath, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        flattened = []
        for item in data:
            flattened.append(np.asarray(item).ravel())
        if flattened:
            return np.concatenate(flattened)
        return np.array([])
    return np.asarray(data).ravel()


def get_labels(particle_type):
    """Generate labels based on particle type."""
    if particle_type == "Bi214":
        return {
            "data": r'$^{214}$Bi - $\beta\gamma$ Data',
            "model1": r'$^{214}$Bi - $\beta\gamma$ MC (Best Fit)',
            "model2": r'$^{214}$Bi - $\beta\gamma$ MC (Initial Fit)',
        }
    elif particle_type == "Po214":
        return {
            "data": r'$^{214}$Po - $\alpha$ Data',
            "model1": r'$^{214}$Po - $\alpha$ MC (Best Fit)',
            "model2": r'$^{214}$Po - $\alpha$ MC (Initial Fit)',
        }
    elif particle_type == "Bi210":
        return {
            "data": r'$^{210}$Bi - $\beta$ Data',
            "model1": r'$^{210}$Bi - $\beta$ MC (Best Fit)',
            "model2": r'$^{210}$Bi - $\beta$ MC (Initial Fit)',
        }
    else:
        raise ValueError(f"Unsupported particle type: {particle_type}")


def calculate_itr(values, label):
    arr = np.asarray(values)
    low = -2.5
    high = 5.0
    mask = (arr >= low) & (arr <= high)
    total = float(len(arr)) if len(arr) else 1.0
    print(f"{label} ITR: {float(np.sum(mask)) / total}")


def plot_comparison(data, model1, model2, output_dir, output_tag, data_label, model1_label, model2_label):
    SNOplus_style()

    binning = np.arange(-5, 350, 1)
    counts_data, bin_edges = np.histogram(data, bins=binning)
    counts_model1, _ = np.histogram(model1, bins=binning)
    counts_model2, _ = np.histogram(model2, bins=binning)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total_data = np.sum(counts_data)
    total_model1 = np.sum(counts_model1)
    total_model2 = np.sum(counts_model2)

    counts_data_norm = counts_data / total_data if total_data else counts_data
    counts_model1_norm = counts_model1 / total_model1 if total_model1 else counts_model1
    counts_model2_norm = counts_model2 / total_model2 if total_model2 else counts_model2

    # Calculate significance for model1: (data - model1) / data
    significance1 = np.zeros_like(bin_centers, dtype=float)
    significance1_err = np.zeros_like(bin_centers, dtype=float)

    for index in range(len(bin_centers)):
        if counts_data[index] > 0 and total_data and total_model1:
            significance1[index] = (counts_data_norm[index] - counts_model1_norm[index]) / counts_data_norm[index]
            err_data_norm = np.sqrt(counts_data[index]) / total_data
            err_model1_norm = np.sqrt(counts_model1[index]) / total_model1
            significance1_err[index] = np.abs(significance1[index]) * np.sqrt(
                (err_data_norm / counts_data_norm[index]) ** 2
                + (err_model1_norm / counts_data_norm[index]) ** 2
            )

    # Calculate significance for model2: (data - model2) / data
    significance2 = np.zeros_like(bin_centers, dtype=float)
    significance2_err = np.zeros_like(bin_centers, dtype=float)

    for index in range(len(bin_centers)):
        if counts_data[index] > 0 and total_data and total_model2:
            significance2[index] = (counts_data_norm[index] - counts_model2_norm[index]) / counts_data_norm[index]
            err_data_norm = np.sqrt(counts_data[index]) / total_data
            err_model2_norm = np.sqrt(counts_model2[index]) / total_model2
            significance2_err[index] = np.abs(significance2[index]) * np.sqrt(
                (err_data_norm / counts_data_norm[index]) ** 2
                + (err_model2_norm / counts_data_norm[index]) ** 2
            )

    mask = counts_data > 0

    def decorate_top_axis(axis):
        axis.legend(loc="upper right", fontsize=18)
        # Place text at top center
        axis.text(0.325, 0.95, "SNO+ Preliminary", transform=axis.transAxes, va="top", ha="center", fontsize=18)
        axis.text(0.325, 0.82, "2.2 g/L PPO + 2.2g/L bis-MSB\nR < 4.0 m", transform=axis.transAxes, va="top", ha="center", fontsize=18)

    # === LINEAR SCALE WITH BOTTOM ===
    fig_lin, (ax_lin_top, ax_lin_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )

    ax_lin_top.errorbar(
        bin_centers[mask],
        counts_data_norm[mask],
        yerr=np.sqrt(counts_data[mask]) / total_data if total_data else None,
        fmt="o",
        color="black",
        linewidth=2,
        label=data_label,
        capsize=2,
    )
    ax_lin_top.step(
        bin_centers,
        counts_model1_norm,
        where="mid",
        color="blue",
        linewidth=2,
        label=model1_label,
    )
    ax_lin_top.step(
        bin_centers,
        counts_model2_norm,
        where="mid",
        color="red",
        linewidth=2,
        label=model2_label,
    )
    ax_lin_top.set_xlim((-5, 100))
    ax_lin_top.set_ylabel("Normalized Counts", labelpad=12)
    ax_lin_top.set_ylim((0, 0.07))
    ax_lin_top.minorticks_on()
    decorate_top_axis(ax_lin_top)

    ax_lin_bottom.errorbar(
        bin_centers[mask],
        significance1[mask],
        yerr=significance1_err[mask],
        fmt="o",
        color="blue",
        label=model1_label,
    )
    ax_lin_bottom.errorbar(
        bin_centers[mask],
        significance2[mask],
        yerr=significance2_err[mask],
        fmt="s",
        color="red",
        label=model2_label,
    )
    ax_lin_bottom.axhline(y=0, color="k", linestyle="--")
    ax_lin_bottom.set_xlabel("Time Residual [ns]", labelpad=12)
    ax_lin_bottom.set_ylabel("(data-MC)/data", labelpad=18)
    ax_lin_bottom.minorticks_on()
    ax_lin_bottom.set_ylim((-1.0, 0.99))

    fig_lin.subplots_adjust(hspace=0.0)
    fig_lin.savefig(output_dir / f"{output_tag}_linear.png", bbox_inches="tight")
    plt.close(fig_lin)

    # === LINEAR SCALE TOP ONLY ===
    fig_lin_top_only, ax_lin_top_only = plt.subplots(
        1, 1,
        figsize=(10, 6),
    )
    
    ax_lin_top_only.errorbar(
        bin_centers[mask],
        counts_data_norm[mask],
        yerr=np.sqrt(counts_data[mask]) / total_data if total_data else None,
        fmt="o",
        color="black",
        linewidth=2,
        label=data_label,
        capsize=2,
    )
    ax_lin_top_only.step(
        bin_centers,
        counts_model1_norm,
        where="mid",
        color="blue",
        linewidth=2,
        label=model1_label,
    )
    ax_lin_top_only.step(
        bin_centers,
        counts_model2_norm,
        where="mid",
        color="red",
        linewidth=2,
        label=model2_label,
    )
    ax_lin_top_only.set_xlim((-5, 100))
    ax_lin_top_only.set_ylabel("Normalized Counts", labelpad=12)
    ax_lin_top_only.set_xlabel("Time Residual [ns]", labelpad=12)
    ax_lin_top_only.set_ylim((0, 0.07))
    ax_lin_top_only.minorticks_on()
    ax_lin_top_only.text(0.325, 0.95, "SNO+ Preliminary", transform=ax_lin_top_only.transAxes, va="top", ha="center", fontsize=18)
    ax_lin_top_only.text(0.325, 0.82, "2.2 g/L PPO + 2.2g/L bis-MSB\nR < 4.0 m", transform=ax_lin_top_only.transAxes, va="top", ha="center", fontsize=18)
    ax_lin_top_only.legend(loc="upper right", fontsize=18)
    
    fig_lin_top_only.subplots_adjust(hspace=0.0)
    fig_lin_top_only.savefig(output_dir / f"{output_tag}_linear_top.png", bbox_inches="tight")
    plt.close(fig_lin_top_only)

    # === LOG SCALE WITH BOTTOM ===
    fig_log, (ax_log_top, ax_log_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )

    ax_log_top.errorbar(
        bin_centers[mask],
        counts_data_norm[mask],
        yerr=np.sqrt(counts_data[mask]) / total_data if total_data else None,
        fmt="o",
        color="black",
        linewidth=2,
        label=data_label,
        capsize=2,
    )
    ax_log_top.step(
        bin_centers,
        counts_model1_norm,
        where="mid",
        color="blue",
        linewidth=2,
        label=model1_label,
    )
    ax_log_top.step(
        bin_centers,
        counts_model2_norm,
        where="mid",
        color="red",
        linewidth=2,
        label=model2_label,
    )
    ax_log_top.set_yscale("log")
    ax_log_top.set_xlim((0, 250))
    ax_log_top.set_ylim((0.99e-5, 0.1))
    ax_log_top.set_ylabel("Normalized Counts", labelpad=12)
    ax_log_top.minorticks_on()
    decorate_top_axis(ax_log_top)

    ax_log_bottom.errorbar(
        bin_centers[mask],
        significance1[mask],
        yerr=significance1_err[mask],
        fmt="o",
        color="blue",
        label=model1_label,
    )
    ax_log_bottom.errorbar(
        bin_centers[mask],
        significance2[mask],
        yerr=significance2_err[mask],
        fmt="s",
        color="red",
        label=model2_label,
    )
    ax_log_bottom.axhline(y=0, color="k", linestyle="--")
    ax_log_bottom.set_xlabel("Time Residual [ns]", labelpad=12)
    ax_log_bottom.set_ylabel("(data-MC)/data", labelpad=18)
    ax_log_bottom.minorticks_on()
    ax_log_bottom.set_ylim((-1.0, 0.99))

    fig_log.subplots_adjust(hspace=0.0)
    fig_log.savefig(output_dir / f"{output_tag}_log.png", bbox_inches="tight")
    plt.close(fig_log)

    # === LOG SCALE TOP ONLY ===
    fig_log_top_only, ax_log_top_only = plt.subplots(
        1, 1,
        figsize=(10, 6),
    )
    
    ax_log_top_only.errorbar(
        bin_centers[mask],
        counts_data_norm[mask],
        yerr=np.sqrt(counts_data[mask]) / total_data if total_data else None,
        fmt="o",
        color="black",
        linewidth=2,
        label=data_label,
        capsize=2,
    )
    ax_log_top_only.step(
        bin_centers,
        counts_model1_norm,
        where="mid",
        color="blue",
        linewidth=2,
        label=model1_label,
    )
    ax_log_top_only.step(
        bin_centers,
        counts_model2_norm,
        where="mid",
        color="red",
        linewidth=2,
        label=model2_label,
    )
    ax_log_top_only.set_yscale("log")
    ax_log_top_only.set_xlim((0, 250))
    ax_log_top_only.set_ylim((0.99e-5, 0.1))
    ax_log_top_only.set_ylabel("Normalized Counts", labelpad=12)
    ax_log_top_only.set_xlabel("Time Residual [ns]", labelpad=12)
    ax_log_top_only.minorticks_on()
    ax_log_top_only.text(0.325, 0.95, "SNO+ Preliminary", transform=ax_log_top_only.transAxes, va="top", ha="center", fontsize=18)
    ax_log_top_only.text(0.325, 0.82, "2.2 g/L PPO + 2.2g/L bis-MSB\nR < 4.0 m", transform=ax_log_top_only.transAxes, va="top", ha="center", fontsize=18)
    ax_log_top_only.legend(loc="upper right", fontsize=18)

    fig_log_top_only.subplots_adjust(hspace=0.0)
    fig_log_top_only.savefig(output_dir / f"{output_tag}_log_top.png", bbox_inches="tight")
    plt.close(fig_log_top_only)


def main():
    parser = argparse.ArgumentParser(description="Compare time residuals from data and two MC ROOT directories.")
    parser.add_argument("datafilepath", help="Path to the data residual file (.npy)")
    parser.add_argument("model1path", help="Path to the first MC RATDS directory containing .root files (Best Fit)")
    parser.add_argument("model2path", help="Path to the second MC RATDS directory containing .root files (Initial Fit)")
    parser.add_argument("particle_type", choices=["Bi214", "Po214", "Bi210"], help="Particle type")
    args = parser.parse_args()

    data = load_residuals(args.datafilepath)
    model1 = extract_residuals(args.model1path, args.particle_type)
    model2 = extract_residuals(args.model2path, args.particle_type)

    labels = get_labels(args.particle_type)
    
    calculate_itr(data, "data")
    calculate_itr(model1, "model1")
    calculate_itr(model2, "model2")

    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tag = f"tres_comparison_{Path(args.datafilepath).stem}_vs_models_{args.particle_type}"
    plot_comparison(data, model1, model2, output_dir, output_tag, labels["data"], labels["model1"], labels["model2"])


if __name__ == "__main__":
    main()
