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
def surrogate_uncert_acquistion(mean, uncertainty, acquisition, meshX, meshY, name_2D,iteration,measured_pts):
    print("In surrogate_uncert_acquistion plotting...")
    """
    Function creates a panel of plots showing 3D and 2D contour plots of the surrogate,
    uncertainty and acquisition functions.
    name_2D is a tuple for the name of projection space 
    measured_pts is the array with measured points in subspace (N,2)  
    """
    SNOplus_style()
    # Override figsize for this plot (keep custom 20x12 instead of style's 10x10)
    matplotlib.rcParams['figure.figsize'] = (20, 12)
    par_name = {
        "0": "T1 [ns]",
        "1": "T2 [ns]",
        "2": "T3 [ns]",
        "3": "T4 [ns]",
        "4": "Rise Time [ns]",
        "5": "A1",
        "6": "A2",
        "7": "A3",
        "8": "A4",
        "9": "bisMSB Reemission Time [ns]",
    } 
    par1_name = par_name[str(name_2D[0])]; par2_name = par_name[str(name_2D[1])]

    # handle when only 1 measured point and array is 1D for imshow heatmap plots
    measured_pts = np.array(measured_pts)
    """
    if measured_pts.size == 2:
        measured_pts = measured_pts[None, :]
    else:
        print("measured_pts ",measured_pts)
    """
    fig_top, top_axes = plt.subplots(nrows=1, ncols=3, subplot_kw={'projection': '3d'})
    fig_top.subplots_adjust(wspace=0.35)

    top_axes[0].plot_surface(meshX, meshY, mean, cmap="inferno")
    top_axes[0].set_title("Surrogate")
    top_axes[0].set_xlabel(par1_name)
    top_axes[0].set_ylabel(par2_name)
    top_axes[0].set_zlabel("Surrogate")

    top_axes[1].plot_surface(meshX, meshY, uncertainty, cmap="inferno")
    top_axes[1].set_title("Uncertainty")
    top_axes[1].set_xlabel(par1_name)
    top_axes[1].set_ylabel(par2_name)
    top_axes[1].set_zlabel("Uncertainty")

    top_axes[2].plot_surface(meshX, meshY, acquisition, cmap="inferno")
    top_axes[2].set_title("Acquisition")
    top_axes[2].set_xlabel(par1_name)
    top_axes[2].set_ylabel(par2_name)
    top_axes[2].set_zlabel("Acquisition")

    fig_bottom, bottom_axes = plt.subplots(nrows=1, ncols=3, figsize=(24,6))
    fig_bottom.subplots_adjust(wspace=0.55)

    img     = bottom_axes[0].imshow(mean, origin="lower", extent=[np.amin(meshX), np.amax(meshX), np.amin(meshY), np.amax(meshY)], aspect="auto", cmap="inferno")
    divider = make_axes_locatable(bottom_axes[0])
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.text(0.5, 1.02, r'$\chi^2$', transform=cbar.ax.transAxes, ha="center", va="bottom")
    bottom_axes[0].set_xlabel(par1_name)
    bottom_axes[0].set_ylabel(par2_name)
    bottom_axes[0].scatter(measured_pts[:-1,0], measured_pts[:-1,1], color="steelblue", s=14, marker="o")
    bottom_axes[0].plot(measured_pts[:,0], measured_pts[:,1], color="steelblue", linestyle="-", linewidth=1.0, marker="")
    bottom_axes[0].scatter(measured_pts[-1,0], measured_pts[-1,1], color="steelblue", s=18, marker="x")

    img     = bottom_axes[1].imshow(uncertainty, origin="lower", extent=[np.amin(meshX), np.amax(meshX), np.amin(meshY), np.amax(meshY)], aspect="auto", cmap="inferno")
    divider = make_axes_locatable(bottom_axes[1])
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.text(0.5, 1.02, r'$\chi^2$', transform=cbar.ax.transAxes, ha="center", va="bottom")
    bottom_axes[1].set_xlabel(par1_name)
    bottom_axes[1].set_ylabel(par2_name)
    bottom_axes[1].scatter(measured_pts[:-1,0], measured_pts[:-1,1], color="steelblue", s=14, marker="o")
    bottom_axes[1].plot(measured_pts[:,0], measured_pts[:,1], color="steelblue", linestyle="-", linewidth=1.0, marker="")
    bottom_axes[1].scatter(measured_pts[-1,0], measured_pts[-1,1], color="steelblue", s=18, marker="x")

    img     = bottom_axes[2].imshow(acquisition, origin="lower", extent=[np.amin(meshX), np.amax(meshX), np.amin(meshY), np.amax(meshY)], aspect="auto", cmap="inferno")
    divider = make_axes_locatable(bottom_axes[2])
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.text(0.5, 1.02, "Expected\nImprovement", transform=cbar.ax.transAxes, ha="center", va="bottom")
    bottom_axes[2].set_xlabel(par1_name)
    bottom_axes[2].set_ylabel(par2_name)
    max_index = np.unravel_index(np.argmax(acquisition), acquisition.shape)
    next_x = meshX[max_index]
    next_y = meshY[max_index]
    bottom_axes[2].scatter(next_x, next_y, color="steelblue", s=60, marker="x", linewidths=2.0)
    os.makedirs(f"results/plots/{iteration}", exist_ok=True)
    fig_top.savefig(f"results/plots/{iteration}/{par1_name}_{par2_name}_upper.pdf", bbox_inches="tight")
    fig_bottom.savefig(f"results/plots/{iteration}/{par1_name}_{par2_name}_lower.pdf", bbox_inches="tight")
    plt.close(fig_top)
    plt.close(fig_bottom)
