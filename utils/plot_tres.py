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

def time_residual_agreement(data, model, iter, pars):
    print("In time_residual_agreement plotting...")
    """
    Function makes a 3 subplot plot showing the binned tRes distributions between data and MC,
    alongside the empirical CDFs of each distribution and max difference between them which
    the ks-test returns.
    """
    """
    data and model are tres, iter represents for the number of iterations
    pars are parameters that are in 1-D array format
    """
    def calculate_cdf(distro):
        """
        Returns the empirical CDF.
        """

        distro = np.sort(distro)

        cdf    = np.arange(1, len(distro)+1) / len(distro)

        return distro, cdf
    
    # find the parameter values used
    parameter_names    = ["T1", "T2", "T3", "T4", "TR", "A1", "A2", "A3", "A4"]
    names              = parameter_names
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 5))
    binning = np.arange(-5, 350, 1)
    values      = [pars[0], pars[1],pars[2],pars[3],pars[4],pars[5],pars[6],pars[7],pars[8]]
    first_line = " | ".join(
        f"{names[i]}: {values[i]:.3f}" for i in range(2)
    )
    second_line = " | ".join(
        f"{names[i]}: {values[i]:.3f}" for i in range(2, 5)
    )
    third_line = " | ".join(
        f"{names[i]}: {values[i]:.3f}" for i in range(5,7)
    )
    fourth_line = " | ".join(
        f"{names[i]}: {values[i]:.3f}" for i in range(7,9)
    )
    

    label_text = first_line + "\n" + second_line + "\n" + third_line + "\n" + fourth_line

    axes[0].plot([], [], linestyle="", label=label_text)
    axes[1].plot([], [], linestyle="", label=label_text)
    
    axes[0].hist(data, bins = binning, density = True, histtype = "step", color = "black", linewidth = 2, label = "data")
    axes[0].hist(model, bins = binning, density = True, histtype = "step", color = "red", linewidth = 2, label = "MC")
    
    axes[0].set_xlim((-5, 100))
    axes[0].set_xlabel("Time Residual [ns]")
    axes[0].legend(loc="upper right")

    axes[1].hist(data, bins = binning, density = True, histtype = "step", color = "black", linewidth = 2, label = "data")
    axes[1].hist(model, bins = binning, density = True, histtype = "step", color = "red", linewidth = 2, label = "MC")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Time Residual [ns]")
    axes[1].legend(loc="upper right")

    x_measured, cdf_measured = calculate_cdf(data)
    x_simulated, cdf_simulated = calculate_cdf(model)

    # Create a common x-axis by combining the unique x-values from both distributions
    x_common = np.sort(np.unique(np.concatenate((x_measured, x_simulated))))

    # Interpolate the CDFs onto the common x-axis
    interp_cdf_measured = interp1d(x_measured, cdf_measured, bounds_error=False, fill_value=(0,1))
    interp_cdf_simulated = interp1d(x_simulated, cdf_simulated, bounds_error=False, fill_value=(0,1))

    cdf_measured_interp = interp_cdf_measured(x_common)
    cdf_simulated_interp = interp_cdf_simulated(x_common)

    # Calculate the KS statistic and find the position of the maximum difference
    ks_statistic = np.max(np.abs(cdf_measured_interp - cdf_simulated_interp))
    max_diff_index = np.argmax(np.abs(cdf_measured_interp - cdf_simulated_interp))

    # Plot CDFs
    axes[2].plot(x_common, cdf_measured_interp, label='Measured Data CDF')
    axes[2].plot(x_common, cdf_simulated_interp, label='Simulated Data CDF')
    axes[2].set_title(f'KS Test: Statistic = {ks_statistic:.3f}')

    # Highlight the maximum difference (KS statistic)
    axes[2].vlines(x_common[max_diff_index], cdf_measured_interp[max_diff_index], cdf_simulated_interp[max_diff_index], 
            colors='r', linestyle='--', label=f'Max Diff (KS statistic)')

    axes[2].set_xlabel('Data Value')
    axes[2].set_ylabel('CDF')
    axes[2].legend()
    
    fig.tight_layout()
    os.makedirs(f"results/plots/{iter}", exist_ok=True)
    plt.savefig(f"results/plots/{iter}/tres_comparison_{iter}.png")
    plt.close()
def threetime_residual_agreement(data, model1,model2):
    print("In time_residual_agreement plotting...")
    """
    Function makes a 2 subplot plot showing the binned tRes distributions between data and MC
    """
    """
    data and model1/model2 are tres, iter represents for the number of iterations
    """

    # find the parameter values used
    parameter_names    = ["T1", "T2", "T3", "T4", "TR", "A1", "A2", "A3", "A4"]
    names              = parameter_names
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
    binning = np.arange(-5, 350, 1)
    
    
    axes[0].hist(data, bins = binning, density = True, histtype = "step", color = "black", linewidth = 2, label = "data")
    axes[0].hist(model1, bins = binning, density = True, histtype = "step", color = "red", linewidth = 2, label = "MC-bayesian")
    axes[0].hist(model2, bins = binning, density = True, histtype = "step", color = "blue", linewidth = 2, label = "MC-gridscan")
    
    axes[0].set_xlim((-5, 100))
    axes[0].set_xlabel("Time Residual [ns]")
    axes[0].legend(loc="upper right")

    axes[1].hist(data, bins = binning, density = True, histtype = "step", color = "black", linewidth = 2, label = "data")
    axes[1].hist(model1, bins = binning, density = True, histtype = "step", color = "red", linewidth = 2, label = "MC-bayesian")
    axes[1].hist(model2, bins = binning, density = True, histtype = "step", color = "blue", linewidth = 2, label = "MC-gridscan")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Time Residual [ns]")
    axes[1].legend(loc="upper right")


    
    
    fig.tight_layout()
    os.makedirs(f"results/plots/", exist_ok=True)
    plt.savefig(f"results/plots/tres_comparison_gridscan_bay.png")
    plt.close()