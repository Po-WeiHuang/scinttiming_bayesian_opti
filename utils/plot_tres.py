import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import ks_2samp
from scipy.interpolate import interp1d
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PlottingStyle'))
from SNOplus_PythonPublicationStyle import SNOplus_style


def time_residual_agreement(data, model, iter, pars):
    print("In time_residual_agreement plotting...")
    """
    Function makes two plots with 2 subplots each:
    1. Normal scale: top (histogram linear), bottom (significance)
    2. Log scale: top (histogram log), bottom (significance)
    Both top and bottom share the x-axis within each plot.
    """
    """
    data and model are tres, iter represents for the number of iterations
    pars are parameters that are in 1-D array format
    """
    SNOplus_style()
    
    def calculate_cdf(distro):
        """
        Returns the empirical CDF.
        """
        distro = np.sort(distro)
        cdf    = np.arange(1, len(distro)+1) / len(distro)
        return distro, cdf
    
    # find the parameter values used
    parameter_names    = ["T1", "T2", "T3", "T4", "TR", "A1", "A2", "A3", "A4","BISRT"]
    names              = parameter_names
    
    binning = np.arange(-5, 350, 1)
    values      = [pars[0], pars[1],pars[2],pars[3],pars[4],pars[5],pars[6],pars[7],pars[8],pars[9]]
    first_line = " | ".join(
        f"{names[i]}: {values[i]:.3f}" for i in range(2)
    )
    second_line = " | ".join(
        f"{names[i]}: {values[i]:.3f}" for i in range(2, 4)
    )
    third_line = " | ".join(
        f"{names[i]}: {values[i]:.3f}" for i in range(5,7)
    )
    fourth_line = " | ".join(
        f"{names[i]}: {values[i]:.3f}" for i in range(7,9)
    )
    fifth_line = " | ".join(
        f"{names[i]}: {values[i]:.3f}" for i in [4,9]
    )
    
    label_text = first_line + "\n" + second_line + "\n" + third_line + "\n" + fourth_line + "\n" + fifth_line
    
    # Compute histograms once
    counts_data, bins_edges = np.histogram(data, bins=binning)
    counts_model, _ = np.histogram(model, bins=binning)
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
    bin_widths = np.diff(bins_edges)
    
    # Normalize counts to density
    total_data = np.sum(counts_data)
    total_model = np.sum(counts_model)
    counts_data_norm = counts_data / total_data
    counts_model_norm = counts_model / total_model
    
    # Calculate significance: (data - mc) / data with Poisson errors
    significance = np.zeros_like(bin_centers)
    significance_err = np.zeros_like(bin_centers)
    
    for i in range(len(bin_centers)):
        if counts_data[i] > 0:
            # (data - mc) / data using normalized counts
            significance[i] = (counts_data_norm[i] - counts_model_norm[i]) / counts_data_norm[i]
            # Error propagation with normalized counts
            # For normalized counts: d(count_norm) = sqrt(count) / total
            err_data_norm = np.sqrt(counts_data[i]) / total_data
            err_model_norm = np.sqrt(counts_model[i]) / total_model
            # For ratio: d(sig) / sig = sqrt((d_data/data)^2 + (d_model/data)^2)
            significance_err[i] = np.abs(significance[i]) * np.sqrt((err_data_norm / counts_data_norm[i])**2 + (err_model_norm / counts_data_norm[i])**2)
        else:
            significance[i] = 0
            significance_err[i] = 0
    
    mask = counts_data > 0  # Only plot where we have data
    """
    # Calculate CDF for KS test (for future use)
    x_measured, cdf_measured = calculate_cdf(data)
    x_simulated, cdf_simulated = calculate_cdf(model)
    x_common = np.sort(np.unique(np.concatenate((x_measured, x_simulated))))
    interp_cdf_measured = interp1d(x_measured, cdf_measured, bounds_error=False, fill_value=(0,1))
    interp_cdf_simulated = interp1d(x_simulated, cdf_simulated, bounds_error=False, fill_value=(0,1))
    cdf_measured_interp = interp_cdf_measured(x_common)
    cdf_simulated_interp = interp_cdf_simulated(x_common)
    ks_statistic = np.max(np.abs(cdf_measured_interp - cdf_simulated_interp))
    """
    # === NORMAL SCALE PLOT ===
    fig_lin, (ax_lin_top, ax_lin_bottom) = plt.subplots(
        2,
        1,
        figsize=None,
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0},
    )
    #print(counts_data[mask][:25])
    #print(counts_data[mask][-25:])
    #print(np.sqrt(counts_data[mask][:25]))
    #print(np.sqrt(counts_data[mask][-25:]))
    #ax_lin_top.plot([], [], linestyle="", label=label_text)
    ax_lin_top.plot([], [], linestyle="")
    ax_lin_top.errorbar(
        bin_centers[mask],
        counts_data_norm[mask],
        yerr=np.sqrt(counts_data[mask])/ total_data,
        fmt="o",
        color="black",
        label="data",
        capsize=2,
    )
    ax_lin_top.hist(
        model,
        bins=binning,
        weights=np.ones_like(model, dtype=float) / total_model,
        histtype="step",
        color="red",
        label="MC",
    )
    ax_lin_top.set_xlim((-5, 100))
    ax_lin_top.set_ylabel("Normalized Counts", labelpad=12)
    ax_lin_top.minorticks_on()
    ax_lin_top.tick_params(which='minor', length=3, width=0.8)
    ax_lin_top.legend(loc="upper right")
    
    ax_lin_bottom.errorbar(bin_centers[mask], significance[mask], yerr=significance_err[mask], 
                           fmt='o', color='black', label='(data-MC)/data')
    ax_lin_bottom.axhline(y=0, color='r', linestyle='--')
    ax_lin_bottom.set_xlabel("Time Residual [ns]", labelpad=12)
    ax_lin_bottom.set_ylabel("(data-MC)/data", labelpad=12)
    ax_lin_bottom.minorticks_on()
    ax_lin_bottom.tick_params(which='minor', length=3, width=0.8)
    ax_lin_bottom.set_ylim((-0.5, 0.49))
    
    fig_lin.subplots_adjust(hspace=0.0)
    os.makedirs(f"results/plots/{iter}", exist_ok=True)
    plt.savefig(f"results/plots/{iter}/tres_comparison_{iter}_linear.png")
    plt.close()
    
    # === LOG SCALE PLOT ===
    binning_log = np.arange(-5, 350, 1)
    counts_data_log, bins_edges_log = np.histogram(data, bins=binning_log)
    counts_model_log, _ = np.histogram(model, bins=binning_log)
    xlim_max = 300
    
    fig_log, (ax_log_top, ax_log_bottom) = plt.subplots(
        2,
        1,
        figsize=None,
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0},
    )
    
    #ax_log_top.plot([], [], linestyle="", label=label_text)
    ax_log_top.plot([], [], linestyle="")
    ax_log_top.errorbar(
        bin_centers[mask],
        counts_data_norm[mask],
        yerr=np.sqrt(counts_data[mask])/ total_data,
        fmt="o",
        color="black",
        label="data",
        capsize=2,
    )
    ax_log_top.hist(
        model,
        bins=binning_log,
        weights=np.ones_like(model, dtype=float) / total_model,
        histtype="step",
        color="red",
        label="MC",
    )
    ax_log_top.set_yscale("log")
    ax_log_top.set_xlim((-5, xlim_max))
    ax_log_top.set_ylim((1e-5, 0.1))
    ax_log_top.set_ylabel("Normalized Counts", labelpad=12)
    ax_log_top.minorticks_on()
    ax_log_top.tick_params(which='minor', length=3, width=0.8)
    ax_log_top.legend(loc="upper right")
    
    ax_log_bottom.errorbar(bin_centers[mask], significance[mask], yerr=significance_err[mask], 
                           fmt='o', color='black', label='(data-MC)/data')
    ax_log_bottom.axhline(y=0, color='r', linestyle='--')
    ax_log_bottom.set_xlabel("Time Residual [ns]", labelpad=12)
    ax_log_bottom.set_ylabel("(data-MC)/data", labelpad=12)
    ax_log_bottom.minorticks_on()
    ax_log_bottom.tick_params(which='minor', length=3, width=0.8)
    ax_log_bottom.set_ylim((-1.0, 0.99))
    
    fig_log.subplots_adjust(hspace=0.0)
    os.makedirs(f"results/plots/{iter}", exist_ok=True)
    plt.savefig(f"results/plots/{iter}/tres_comparison_{iter}_log.png")
    plt.close()
def threetime_residual_agreement(data, model1,model2,label1,label2):
    print("In time_residual_agreement plotting...")
    SNOplus_style()
    """
    Function makes separate plots for linear and log scales showing the binned tRes distributions between data and MC
    with significance panels for both models
    """
    """
    data and model1/model2 are tres
    """

    binning = np.arange(-5, 350, 1)
    
    # ITR calculation
    def ITR(data,type="data"):
        arrdata = np.array(data) 
        low = -2.5; high = 5.
        mask = (arrdata >= low) & (arrdata <= high)
        count_in_range = float(np.sum(mask))
        total_count = float(len(data))
        print(f"{type} ITR: {count_in_range/total_count}")
    
    ITR(data,"data")
    ITR(model1,"label1")
    ITR(model2,"label2")
    
    os.makedirs(f"results/plots/", exist_ok=True)
    
    # Calculate histograms and significance for both models
    counts_data, bin_edges = np.histogram(data, bins=binning)
    counts_model1, _ = np.histogram(model1, bins=binning)
    counts_model2, _ = np.histogram(model2, bins=binning)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    total_data = np.sum(counts_data)
    total_model1 = np.sum(counts_model1)
    total_model2 = np.sum(counts_model2)
    
    

    counts_data_norm = counts_data / total_data
    counts_model1_norm = counts_model1 / total_model1
    counts_model2_norm = counts_model2 / total_model2

    
    
    # Calculate significance for model1: (data - model1) / data with Poisson errors
    significance1 = np.zeros_like(bin_centers)
    significance1_err = np.zeros_like(bin_centers)
    
    for i in range(len(bin_centers)):
        if counts_data[i] > 0:
            significance1[i] = (counts_data_norm[i] - counts_model1_norm[i]) / counts_data_norm[i]
            err_data_norm = np.sqrt(counts_data[i]) / total_data
            err_model1_norm = np.sqrt(counts_model1[i]) / total_model1
            significance1_err[i] = np.abs(significance1[i]) * np.sqrt((err_data_norm / counts_data_norm[i])**2 + (err_model1_norm / counts_data_norm[i])**2)
        else:
            significance1[i] = 0
            significance1_err[i] = 0
    
    # Calculate significance for model2: (data - model2) / data with Poisson errors
    significance2 = np.zeros_like(bin_centers)
    significance2_err = np.zeros_like(bin_centers)

    
    
    for i in range(len(bin_centers)):
        if counts_data[i] > 0:
            significance2[i] = (counts_data_norm[i] - counts_model2_norm[i]) / counts_data_norm[i]
            err_data_norm = np.sqrt(counts_data[i]) / total_data
            err_model2_norm = np.sqrt(counts_model2[i]) / total_model2
            significance2_err[i] = np.abs(significance2[i]) * np.sqrt((err_data_norm / counts_data_norm[i])**2 + (err_model2_norm / counts_data_norm[i])**2)
        else:
            significance2[i] = 0
            significance2_err[i] = 0

    

    mask = counts_data > 0  # Only plot where we have data
    
    # === LINEAR SCALE PLOT ===
    fig_lin, (ax_lin_top, ax_lin_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0},
    )
   
    ax_lin_top.errorbar(
        bin_centers[mask],
        counts_data_norm[mask],
        yerr=np.sqrt(counts_data[mask])/ total_data,
        fmt='o',
        color='black',
        linewidth=2,
        label='data',
        capsize=2,
    )
    ax_lin_top.hist(model1, bins=binning, weights=np.ones_like(model1, dtype=float) / total_model1, histtype="step", color="red", linewidth=2, label=label1)
    ax_lin_top.hist(model2, bins=binning, weights=np.ones_like(model2, dtype=float) / total_model2, histtype="step", color="blue", linewidth=2, label=label2)
    ax_lin_top.set_xlim((-5, 100))
    ax_lin_top.set_ylabel("Normalized Counts", labelpad=12)
    ax_lin_top.minorticks_on()
    ax_lin_top.tick_params(which='minor', length=3, width=0.8)
    ax_lin_top.legend(loc="upper right")
    
    ax_lin_bottom.errorbar(bin_centers[mask], significance1[mask], yerr=significance1_err[mask], 
                           fmt='o', color='red', label=label1)
    ax_lin_bottom.errorbar(bin_centers[mask], significance2[mask], yerr=significance2_err[mask], 
                           fmt='s', color='blue', label=label2)
    ax_lin_bottom.axhline(y=0, color='k', linestyle='--')
    ax_lin_bottom.set_xlabel("Time Residual [ns]", labelpad=12)
    ax_lin_bottom.set_ylabel("(data-MC)/data", labelpad=12)
    ax_lin_bottom.minorticks_on()
    ax_lin_bottom.tick_params(which='minor', length=3, width=0.8)
    ax_lin_bottom.set_ylim((-0.5, 0.49))
    
    fig_lin.subplots_adjust(hspace=0.0)
    plt.savefig(f"results/plots/tres_comparison_gridscan_bay_linear.png")
    plt.close()
    
    # === LOG SCALE PLOT ===
    fig_log, (ax_log_top, ax_log_bottom) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0},
    )
    
    ax_log_top.errorbar(
        bin_centers[mask],
        counts_data_norm[mask],
        yerr=np.sqrt(counts_data[mask])/ total_data,
        fmt='o',
        color='black',
        linewidth=2,
        label='data',
        capsize=2,
    )
    ax_log_top.hist(model1, bins=binning, weights=np.ones_like(model1, dtype=float) / total_model1, histtype="step", color="red", linewidth=2, label=label1)
    ax_log_top.hist(model2, bins=binning, weights=np.ones_like(model2, dtype=float) / total_model2, histtype="step", color="blue", linewidth=2, label=label2)
    ax_log_top.set_xlim((-5, 300))
    ax_log_top.set_yscale("log")
    ax_log_top.set_ylabel("Normalized Counts", labelpad=12)
    ax_log_top.minorticks_on()
    ax_log_top.tick_params(which='minor', length=3, width=0.8)
    ax_log_top.legend(loc="upper right")
    
    ax_log_bottom.errorbar(bin_centers[mask], significance1[mask], yerr=significance1_err[mask], 
                           fmt='o', color='red', label=label1)
    ax_log_bottom.errorbar(bin_centers[mask], significance2[mask], yerr=significance2_err[mask], 
                           fmt='s', color='blue', label=label2)
    ax_log_bottom.axhline(y=0, color='k', linestyle='--')
    ax_log_bottom.set_xlabel("Time Residual [ns]", labelpad=12)
    ax_log_bottom.set_ylabel("(data-MC)/data", labelpad=12)
    ax_log_bottom.minorticks_on()
    ax_log_bottom.tick_params(which='minor', length=3, width=0.8)
    ax_log_bottom.set_ylim((-1.0, 0.99))
    
    fig_log.subplots_adjust(hspace=0.0)
    plt.savefig(f"results/plots/tres_comparison_gridscan_bay_log.png")
    plt.close()