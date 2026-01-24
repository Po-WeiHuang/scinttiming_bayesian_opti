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
def surrogate_uncert_acquistion(mean, uncertainty, acquisition, meshX, meshY, name_2D,iteration,measured_pts):
    print("In surrogate_uncert_acquistion plotting...")
    """
    Function creates a panel of plots showing 3D and 2D contour plots of the surrogate,
    uncertainty and acquisition functions.
    name_2D is a tuple for the name of projection space 
    measured_pts is the array with measured points in subspace (N,2)  
    """
    par_name = {
        "0": "T1",
        "1": "T2",
        "2": "T3",
        "3": "T4",
        "4": "TR",
        "5": "A1",
        "6": "A2",
        "7": "A3",
        "8": "A4",
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
    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (20, 12))
    for i in range(3):
        axes[0, i].remove()  # Remove the existing 2D subplot placeholder
        axes[0, i] = fig.add_subplot(2, 3, i+1, projection='3d')

    axes[0,0].plot_surface(meshX, meshY, mean, cmap = "inferno")
    axes[0,0].set_title("Surrogate")
    axes[0,0].set_xlabel(par1_name)
    axes[0,0].set_ylabel(par2_name)
    axes[0,0].set_zlabel("Surrogate")

    axes[0,1].plot_surface(meshX, meshY, uncertainty, cmap = "inferno")
    axes[0,1].set_title("Uncertainty")
    axes[0,1].set_xlabel(par1_name)
    axes[0,1].set_ylabel(par2_name)
    axes[0,1].set_zlabel("Uncertainty")

    axes[0,2].plot_surface(meshX, meshY, acquisition, cmap = "inferno")
    axes[0,2].set_title("Acquisition")
    axes[0,2].set_xlabel(par1_name)
    axes[0,2].set_ylabel(par2_name)
    axes[0,2].set_zlabel("acquisition")

    img     = axes[1,0].imshow(mean, origin = "lower", extent = [np.amin(meshX), np.amax(meshX), np.amin(meshY), np.amax(meshY)], aspect = "auto", cmap = "inferno")
    divider = make_axes_locatable(axes[1,0])
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax = cax)
    axes[1,0].set_xlabel(par1_name)
    axes[1,0].set_ylabel(par2_name)
    axes[1,0].scatter(measured_pts[:-1,0], measured_pts[:-1,1], color = "red", marker = "o")
    axes[1,0].plot(measured_pts[:,0], measured_pts[:,1], color = "red", linestyle = "--", marker = "")
    axes[1,0].scatter(measured_pts[-1,0], measured_pts[-1,1], color = "red", marker = "x")

    img     = axes[1,1].imshow(uncertainty, origin = "lower", extent = [np.amin(meshX), np.amax(meshX),np.amin(meshY), np.amax(meshY)], aspect = "auto", cmap = "inferno")
    divider = make_axes_locatable(axes[1,1])
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax = cax)
    axes[1,1].set_xlabel(par1_name)
    axes[1,1].set_ylabel(par2_name)
    axes[1,1].scatter(measured_pts[:-1,0], measured_pts[:-1,1], color = "red", marker = "o")
    axes[1,1].plot(measured_pts[:,0], measured_pts[:,1], color = "red", linestyle = "--", marker = "")
    axes[1,1].scatter(measured_pts[-1,0], measured_pts[-1,1], color = "red", marker = "x")

    img     = axes[1,2].imshow(acquisition, origin = "lower", extent = [np.amin(meshX), np.amax(meshX), np.amin(meshY), np.amax(meshY)], aspect = "auto", cmap = "inferno")
    divider = make_axes_locatable(axes[1,2])
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax = cax)
    axes[1,2].set_xlabel(par1_name)
    axes[1,2].set_ylabel(par2_name)
    os.makedirs(f"results/plots/{iteration}", exist_ok=True)
    plt.savefig(f"results/plots/{iteration}/{par1_name}_{par2_name}.pdf")
    plt.close()
