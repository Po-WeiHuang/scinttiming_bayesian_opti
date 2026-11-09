"""Radial position bias (median) and resolution (68% half-width) of tagged BiPo214 MC events, in 1 m shells.
Uncertainties are bootstrap standard deviations (--n-boot resamples, default 1000).

    bias = posr_av - sqrt(Bi_mcPosx**2 + Bi_mcPosy**2 + (Bi_mcPosz - (posZ - posz_av))**2)

posZ - posz_av is the AV offset, so the square root is the MC-truth radius in AV coordinates.
The truth vertex comes from the Bi_mcPos* branches for both particles: Bi and Po decay at the
same point, Bi_mcPos* is identical in BiT and PoT, and Po_mcPosx is unusable (the tagging code
fills it with mcPosz).

Example:
    python utils/plot_posbias.py \
        "/data/snoplus3/weiii/BiPo214/fullFill/rat-8.3.3/bismsb/mcNtuple/mc_ScintFit_2p2ppo_2p2bismsbBipo214_833/*.ntuple.root" \
        Bi214 --phase bismsb --version rat-8.3.3 --output-dir results/plots/fullFill_data_mc
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PlottingStyle"))
from SNOplus_PythonPublicationStyle import SNOplus_style

TREE_BY_PARTICLE = {"Bi214": "BiT", "Po214": "PoT"}
PARTICLE_LABEL = {"Bi214": r"$^{214}$Bi - $\beta\gamma$", "Po214": r"$^{214}$Po - $\alpha$"}
PHASE_TEXT = {"ppo": "2.2 g/L PPO", "bismsb": "2.2 g/L PPO + 2.2g/L bis-MSB"}
BRANCHES = ["posr_av", "posZ", "posz_av", "Bi_mcPosx", "Bi_mcPosy", "Bi_mcPosz"]

# Ordinal single-hue ramp (light -> dark = inner -> outer shell); line styles give a second encoding.
SHELL_COLORS = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SHELL_STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]


def load(pattern, tree):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files match {pattern}")
    sources = []
    for f in files:
        with uproot.open(f) as r:
            if tree in r and r[tree].num_entries > 0:
                sources.append(f"{f}:{tree}")
    print(f"{len(sources)}/{len(files)} files have entries in {tree}")
    return uproot.concatenate(sources, BRANCHES, library="np")


def position_bias(a):
    av_offset = a["posZ"] - a["posz_av"]
    true_r = np.sqrt(a["Bi_mcPosx"] ** 2 + a["Bi_mcPosy"] ** 2 + (a["Bi_mcPosz"] - av_offset) ** 2)
    return a["posr_av"] - true_r, true_r


def has_truth(a):
    """Real data carries mcPos = 0 for every event; require a spread of truth vertices."""
    return np.ptp(a["Bi_mcPosx"]) > 0 and np.ptp(a["Bi_mcPosz"]) > 0


def median_width(x):
    q16, q50, q84 = np.percentile(x, [16, 50, 84])
    return q50, (q84 - q16) / 2


def shell_stats(bias, radius, edges, n_boot, rng):
    """Per shell: N, median bias, 68% half-width, and their bootstrap (n_boot resamples) standard deviations."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = bias[(radius >= lo) & (radius < hi)]
        if len(b) < 2:
            rows.append((lo, hi, len(b), np.nan, np.nan, np.nan, np.nan))
            continue
        med, w68 = median_width(b)
        boot = np.array([median_width(b[rng.integers(0, len(b), len(b))]) for _ in range(n_boot)])
        rows.append((lo, hi, len(b), med, boot[:, 0].std(ddof=1), w68, boot[:, 1].std(ddof=1)))
    return rows


def compute(pattern, args, rng, label):
    a = load(pattern, TREE_BY_PARTICLE[args.particle])
    if not has_truth(a):
        raise SystemExit(f"{label}: Bi_mcPos* is constant (all {a['Bi_mcPosx'][0]:g}) - no MC truth, so "
                         "posr_av - R_true is not defined for this input.")
    bias, true_r = position_bias(a)
    radius = true_r if args.shell_by == "true" else a["posr_av"]
    edges = np.arange(args.rmin, args.rmax + 0.5 * args.rstep, args.rstep)
    rows = shell_stats(bias, radius, edges, args.n_boot, rng)
    print(f"--- {label}: shell  N  median_bias+-boot  68%_halfwidth+-boot  [mm]")
    for lo, hi, n, med, emed, w, ew in rows:
        print(f"  {lo/1000:.0f}-{hi/1000:.0f} m {n:8d}  {med:7.2f} +- {emed:5.2f}   {w:7.2f} +- {ew:5.2f}")
    return bias, radius, edges, rows


def decorate(ax, args, x=0.03):
    ax.text(x, 0.95, "SNO+ Simulation", transform=ax.transAxes, va="top", ha="left", fontsize=18)
    ax.text(x, 0.85, f"{PHASE_TEXT[args.phase]}\n{PARTICLE_LABEL[args.particle]}", transform=ax.transAxes,
            va="top", ha="left", fontsize=16)


def main():
    p = argparse.ArgumentParser(description="Plot radial position bias (median) and resolution (68% half-width) "
                                            "of tagged BiPo214 MC events in radial shells.")
    p.add_argument("input", help="Tagged MC ntuple glob (quote it) or directory")
    p.add_argument("particle", choices=sorted(TREE_BY_PARTICLE), help="Bi214 (BiT tree) or Po214 (PoT tree)")
    p.add_argument("--phase", choices=sorted(PHASE_TEXT), required=True)
    p.add_argument("--version", default="", help="RAT version label of the MC, e.g. rat-8.3.3")
    p.add_argument("--shell-by", choices=["true", "reco"], default="true",
                   help="Radius that defines the shell: MC-truth radius (default) or reconstructed posr_av")
    p.add_argument("--rmin", type=float, default=1000.0, help="Inner edge of first shell [mm]")
    p.add_argument("--rmax", type=float, default=6000.0, help="Outer edge of last shell [mm]")
    p.add_argument("--rstep", type=float, default=1000.0, help="Shell width [mm]")
    p.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples for the uncertainties")
    p.add_argument("--seed", type=int, default=12345, help="Bootstrap random seed")
    p.add_argument("--bias-range", type=float, nargs=2, default=(-600.0, 600.0), help="Histogram range [mm]")
    p.add_argument("--bias-bin", type=float, default=20.0, help="Histogram bin width [mm]")
    p.add_argument("--output-dir", default="results/plots")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    as_pattern = lambda x: os.path.join(x, "*.root") if os.path.isdir(x) else x
    samples = [("MC", args.version, "#256abf", compute(as_pattern(args.input), args, rng, "MC"))]
    radius_name = "true R" if args.shell_by == "true" else "reco R"

    SNOplus_style()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"posbias_{args.phase}_{args.particle}" + (f"_{args.version}" if args.version else "") + f"_shellby{args.shell_by}"

    # Bias distribution per shell for the MC sample (area-normalised)
    bias, radius, edges, rows = samples[0][3]
    bins = np.arange(args.bias_range[0], args.bias_range[1] + args.bias_bin, args.bias_bin)
    fig, ax = plt.subplots(figsize=(10, 8))
    for k, (lo, hi, n, med, *_rest) in enumerate(rows):
        if n == 0:
            continue
        sel = (radius >= lo) & (radius < hi)
        ax.hist(bias[sel], bins=bins, density=True, histtype="step", linewidth=2,
                color=SHELL_COLORS[k % len(SHELL_COLORS)], linestyle=SHELL_STYLES[k % len(SHELL_STYLES)],
                label=f"{lo/1000:.0f}-{hi/1000:.0f} m (N={n}, median={med:+.0f} mm)")
    ax.axvline(0, color="0.6", linewidth=1, zorder=0)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.set_xlabel(r"$R_{\mathrm{reco}} - R_{\mathrm{true}}$ [mm]", labelpad=12)
    ax.set_ylabel("Normalized Counts / mm", labelpad=12)
    ax.minorticks_on()
    ax.legend(loc="upper right", fontsize=14, title=f"Shell in {radius_name}", title_fontsize=14)
    decorate(ax, args)
    ax.text(0.03, 0.70, f"MC, {args.version}", transform=ax.transAxes, va="top", ha="left", fontsize=16)
    fig.savefig(out / f"{tag}_shells.png", bbox_inches="tight")
    plt.close(fig)

    # Summary: median bias (top) and 68% half-width (bottom) per shell, bootstrap uncertainties
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.0})
    xerr = args.rstep / 2000
    for name, ver, color, (_b, _r, edges, rows) in samples:
        centers = np.array([(lo + hi) / 2000 for lo, hi, *_ in rows])
        med = np.array([r[3] for r in rows]); emed = np.array([r[4] for r in rows])
        w68 = np.array([r[5] for r in rows]); ew68 = np.array([r[6] for r in rows])
        label = f"{name}, {ver}" if ver else name
        for ax, val, err in ((ax1, med, emed), (ax2, w68, ew68)):
            # shell width as a thin horizontal bar; bootstrap uncertainty as a capped vertical bar drawn above a small marker
            ax.errorbar(centers, val, xerr=xerr, fmt="none", ecolor=color, elinewidth=1.2, alpha=0.6)
            ax.errorbar(centers, val, yerr=err, fmt="o", markersize=4, color=color, ecolor=color,
                        elinewidth=2, capsize=6, capthick=2, barsabove=True, label=label if ax is ax1 else None)
    all_rows = [r for *_x, (_b, _r, _e, rows) in samples for r in rows]
    # fit each y-axis to its points so the sub-mm bootstrap bars are resolvable
    blo = min(0.0, np.nanmin([r[3] - r[4] for r in all_rows])); bhi = np.nanmax([r[3] + r[4] for r in all_rows])
    wlo = np.nanmin([r[5] - r[6] for r in all_rows]); whi = np.nanmax([r[5] + r[6] for r in all_rows])
    ax1.axhline(0, color="0.6", linewidth=1, zorder=0)
    ax1.set_ylabel("Bias [mm]", labelpad=12)
    ax1.set_ylim(blo - 3, bhi + 0.9 * (bhi - blo) + 3)   # headroom for the label block
    ax1.minorticks_on()
    ax1.legend(loc="upper right", fontsize=16)
    decorate(ax1, args)
    ax2.set_ylabel("Resolution [mm]", labelpad=12)
    ax2.set_ylim(wlo - 0.35 * (whi - wlo) - 2, whi + 0.35 * (whi - wlo) + 2)
    ax2.set_xlabel(f"Shell {radius_name} [m]", labelpad=12)
    ax2.set_xlim(edges[0] / 1000, edges[-1] / 1000)
    ax2.minorticks_on()
    ax2.text(0.97, 0.06, f"uncertainties: bootstrap, {args.n_boot} resamples", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=13, color="0.35")
    fig.savefig(out / f"{tag}_summary.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out / (tag + '_shells.png')} and {out / (tag + '_summary.png')}")


if __name__ == "__main__":
    main()
