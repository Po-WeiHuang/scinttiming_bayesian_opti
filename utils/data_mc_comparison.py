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


PHASE_TEXT = {
    "ppo": "2.2 g/L PPO\nR < 4.0 m",
    "bismsb": "2.2 g/L PPO + 2.2g/L bis-MSB\nR < 4.0 m",
}

CHUNK_SIZE = 50_000_000


def load_residuals(filepath):
    # Plain numeric arrays are memory-mapped so multi-GB MC files are not read into RAM at once.
    data = np.load(filepath, mmap_mode="r", allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        flattened = []
        for item in data:
            flattened.append(np.asarray(item).ravel())
        if flattened:
            return np.concatenate(flattened)
        return np.array([])
    return data.ravel() if data.ndim != 1 else data


def chunked_histogram(values, bins):
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    for start in range(0, len(values), CHUNK_SIZE):
        chunk_counts, _ = np.histogram(np.asarray(values[start:start + CHUNK_SIZE]), bins=bins)
        counts += chunk_counts
    return counts


def particle_labels(particle_type, data_version=None, mc_version=None):
    if particle_type == "Bi214":
        base = r'$^{214}$Bi - $\beta\gamma$'
    elif particle_type == "Po214":
        base = r'$^{214}$Po - $\alpha$'
    else:
        raise ValueError(f"Unsupported particle type: {particle_type}")
    data_label = f"{base} Data" + (f", {data_version}" if data_version else "")
    mc_label = f"{base} MC" + (f", {mc_version}" if mc_version else "")
    return data_label, mc_label


def calculate_itr(values, label):
    low = -2.5
    high = 5.0
    in_window = 0
    for start in range(0, len(values), CHUNK_SIZE):
        chunk = np.asarray(values[start:start + CHUNK_SIZE])
        in_window += int(np.count_nonzero((chunk >= low) & (chunk <= high)))
    total = float(len(values)) if len(values) else 1.0
    print(f"{label} ITR: {in_window / total}")


def plot_comparison(data, model, output_dir, output_tag, particle_type,
                    data_version=None, mc_version=None, phase="bismsb"):
    SNOplus_style()

    data_label, model_label = particle_labels(particle_type, data_version, mc_version)
    phase_text = PHASE_TEXT[phase]

    binning = np.arange(-5, 350, 1)
    counts_data = chunked_histogram(data, binning)
    counts_model = chunked_histogram(model, binning)
    bin_edges = binning
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total_data = np.sum(counts_data)
    total_model = np.sum(counts_model)

    counts_data_norm = counts_data / total_data if total_data else counts_data
    counts_model_norm = counts_model / total_model if total_model else counts_model

    significance = np.zeros_like(bin_centers, dtype=float)
    significance_err = np.zeros_like(bin_centers, dtype=float)

    for index in range(len(bin_centers)):
        if counts_data[index] > 0 and total_data and total_model:
            significance[index] = (counts_data_norm[index] - counts_model_norm[index]) / counts_data_norm[index]
            err_data_norm = np.sqrt(counts_data[index]) / total_data
            err_model_norm = np.sqrt(counts_model[index]) / total_model
            significance_err[index] = np.abs(significance[index]) * np.sqrt(
                (err_data_norm / counts_data_norm[index]) ** 2
                + (err_model_norm / counts_data_norm[index]) ** 2
            )

    mask = counts_data > 0

    def decorate_top_axis(axis):
        #axis.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.0)
        #axis.tick_params(axis="both", which="minor", length=3, width=0.8)
        axis.legend(loc="upper right", fontsize=18)
        # Place text at top center
        axis.text(0.45, 0.95, "SNO+ Preliminary", transform=axis.transAxes, va="top", ha="center", fontsize=18)
        axis.text(0.45, 0.82, phase_text, transform=axis.transAxes, va="top", ha="center", fontsize=18)

    # === LINEAR SCALE WITH BOTTOM ===
    fig_lin, (ax_lin_top, ax_lin_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )

    ax_lin_top.step(
        bin_centers,
        counts_model_norm,
        where="mid",
        color="blue",
        linewidth=2,
        label=model_label,
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
    ax_lin_top.set_xlim((-5, 100))
    ax_lin_top.set_ylabel("Normalized Counts", labelpad=12)
    ax_lin_top.set_ylim((0, 0.055))
    ax_lin_top.minorticks_on()
    decorate_top_axis(ax_lin_top)

    ax_lin_bottom.errorbar(
        bin_centers[mask],
        significance[mask],
        yerr=significance_err[mask],
        fmt="o",
        color="black",
        label="(data-MC)/data",
    )
    ax_lin_bottom.axhline(y=0, color="r", linestyle="--")
    ax_lin_bottom.set_xlabel("Time Residual [ns]", labelpad=12)
    ax_lin_bottom.set_ylabel("(data-MC)/data", labelpad=18)
    ax_lin_bottom.minorticks_on()
    #ax_lin_bottom.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.0)
    #ax_lin_bottom.tick_params(axis="both", which="minor", length=3, width=0.8)
    ax_lin_bottom.set_ylim((-1.0, 0.99))

    fig_lin.subplots_adjust(hspace=0.0)
    fig_lin.savefig(output_dir / f"{output_tag}_linear.png", bbox_inches="tight")
    plt.close(fig_lin)

    # === LINEAR SCALE TOP ONLY ===
    fig_lin_top_only, ax_lin_top_only = plt.subplots(
        1, 1,
        figsize=(10, 6),
    )
    
    ax_lin_top_only.step(
        bin_centers,
        counts_model_norm,
        where="mid",
        color="blue",
        linewidth=2,
        label=model_label,
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
    ax_lin_top_only.set_xlim((-5, 100))
    ax_lin_top_only.set_ylabel("Normalized Counts", labelpad=12)
    ax_lin_top_only.set_xlabel("Time Residual [ns]", labelpad=12)
    ax_lin_top_only.set_ylim((0, 0.055))
    ax_lin_top_only.minorticks_on()
    #ax_lin_top_only.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.0)
    #ax_lin_top_only.tick_params(axis="both", which="minor", length=3, width=0.8)
    ax_lin_top_only.text(0.45, 0.95, "SNO+ Preliminary", transform=ax_lin_top_only.transAxes, va="top", ha="center", fontsize=18)
    ax_lin_top_only.text(0.45, 0.82, phase_text, transform=ax_lin_top_only.transAxes, va="top", ha="center", fontsize=18)
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
        counts_model_norm,
        where="mid",
        color="blue",
        linewidth=2,
        label=model_label,
    )
    ax_log_top.set_yscale("log")
    ax_log_top.set_xlim((0, 250))
    ax_log_top.set_ylim((0.99e-5, 0.1))
    ax_log_top.set_ylabel("Normalized Counts", labelpad=12)
    ax_log_top.minorticks_on()
    decorate_top_axis(ax_log_top)

    ax_log_bottom.errorbar(
        bin_centers[mask],
        significance[mask],
        yerr=significance_err[mask],
        fmt="o",
        color="black",
        label="(data-MC)/data",
    )
    ax_log_bottom.axhline(y=0, color="r", linestyle="--")
    ax_log_bottom.set_xlabel("Time Residual [ns]", labelpad=12)
    ax_log_bottom.set_ylabel("(data-MC)/data", labelpad=18)
    ax_log_bottom.minorticks_on()
    #ax_log_bottom.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.0)
    #ax_log_bottom.tick_params(axis="both", which="minor", length=3, width=0.8)
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
        counts_model_norm,
        where="mid",
        color="blue",
        linewidth=2,
        label=model_label,
    )
    ax_log_top_only.set_yscale("log")
    ax_log_top_only.set_xlim((0, 250))
    ax_log_top_only.set_ylim((0.99e-5, 0.1))
    ax_log_top_only.set_ylabel("Normalized Counts", labelpad=12)
    ax_log_top_only.set_xlabel("Time Residual [ns]", labelpad=12)
    ax_log_top_only.minorticks_on()
    #ax_log_top_only.tick_params(axis="both", which="major", labelsize=18, length=6, width=1.0)
    #ax_log_top_only.tick_params(axis="both", which="minor", length=3, width=0.8)
    ax_log_top_only.text(0.45, 0.95, "SNO+ Preliminary", transform=ax_log_top_only.transAxes, va="top", ha="center", fontsize=18)
    ax_log_top_only.text(0.45, 0.82, phase_text, transform=ax_log_top_only.transAxes, va="top", ha="center", fontsize=18)
    ax_log_top_only.legend(loc="upper right", fontsize=18)

    fig_log_top_only.subplots_adjust(hspace=0.0)
    fig_log_top_only.savefig(output_dir / f"{output_tag}_log_top.png", bbox_inches="tight")
    plt.close(fig_log_top_only)


def main():
    parser = argparse.ArgumentParser(description="Compare time residuals from one data file and one MC sample.")
    parser.add_argument("datafilepath", help="Path to the data residual file (.npy)")
    parser.add_argument("mcmodelfilepath", help="MC residual file (.npy), or an MC RATDS directory containing .root files")
    parser.add_argument("particle_type", choices=["Bi214", "Po214"], help="Particle type used for MC extraction and labels")
    parser.add_argument("--phase", choices=sorted(PHASE_TEXT), default="bismsb", help="Detector phase; sets the cocktail text and output name")
    parser.add_argument("--data-version", default=None, help="RAT version appended to the data legend entry, e.g. rat-8.3.1")
    parser.add_argument("--mc-version", default=None, help="RAT version appended to the MC legend entry, e.g. rat-8.3.3")
    parser.add_argument("--output-dir", default="results/plots", help="Directory for the output PNGs")
    args = parser.parse_args()

    data = load_residuals(args.datafilepath)
    if args.mcmodelfilepath.endswith(".npy"):
        model = load_residuals(args.mcmodelfilepath)
    else:
        # Imported lazily: it pulls in RAT/ROOT, which the .npy path does not need.
        from twomodels_comparison import extract_residuals
        model = extract_residuals(args.mcmodelfilepath, args.particle_type)

    print(f"data entries: {len(data)}, mc entries: {len(model)}")
    calculate_itr(data, "data")
    calculate_itr(model, "mc")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    versions = "_".join(v for v in (args.data_version, args.mc_version) if v)
    output_tag = f"tres_comparison_{args.phase}_{args.particle_type}" + (f"_{versions}" if versions else "")
    plot_comparison(data, model, output_dir, output_tag, args.particle_type,
                    data_version=args.data_version, mc_version=args.mc_version, phase=args.phase)


if __name__ == "__main__":
    main()