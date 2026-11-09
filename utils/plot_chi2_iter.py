import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import sys
import os
import argparse
from matplotlib.ticker import ScalarFormatter
sys.path.insert(0, "/home/huangp/pytor_bayesian_opti")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PlottingStyle'))
from SNOplus_PythonPublicationStyle import SNOplus_style 
if __name__ == "__main__":
    SNOplus_style()
    parser = argparse.ArgumentParser(description="Plot chi2 iteration history.")
    parser.add_argument(
        "par_dir",
        nargs="?",
        default="results/pars",
        help="Directory containing the .npy parameter/result files",
    )
    args = parser.parse_args()

    with open("paramsbound.json", "r") as f:
        bounds = json.load(f)
    particle_type = bounds.get("Type", None)
    particle_label_map = {
        "Po214": r"$^{214}$Po - $\alpha$",
        "Bi214": r"$^{214}$Bi - $\beta\gamma$",
        "Bi210": r"$^{210}$Bi - $\beta$",
    }
    particle_label = particle_label_map.get(particle_type, "") if particle_type else ""
    with open("currentparams.json", "r") as f:
        params = json.load(f)
    finaliteration = int(params["iter"])
    iteration = np.array([i for i in range(1,finaliteration+1)])
    # Loading all .npy
    dir_path = Path(args.par_dir)
    npy_files = sorted(
        [p for p in dir_path.glob("*.npy") if p.stem.isdigit()],
        key=lambda path: int(path.stem),
    )
    arrays = [np.load(p) for p in npy_files]
    # Split to parameter_set & objective
    train = np.vstack(arrays)
    train_Y = train[:,-2]



    fig,ax = plt.subplots(figsize=(10, 6))
    ax.plot(iteration,train_Y,'o-',color="blue", linewidth=2, markersize=4)
    ax.minorticks_on()
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"$\chi^2$", labelpad=5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    chi2_formatter = ScalarFormatter(useMathText=True)
    chi2_formatter.set_powerlimits((3, 3))
    ax.yaxis.set_major_formatter(chi2_formatter)
    ax.text(0.65, 0.95, 'SNO+ Preliminary', transform=ax.transAxes, va='top', ha='center', fontsize=20)
    if particle_label:
        ax.text(0.65, 0.82, particle_label, transform=ax.transAxes, va='top', ha='center', fontsize=20)
        #ax.text(0.65, 0.69, '2.2 g/L PPO + 2.2g/L bis-MSB\nR < 4.0 m', transform=ax.transAxes, va='top', ha='center', fontsize=20)
        ax.text(0.65, 0.69, '2.2 g/L PPO \nR < 4.0 m', transform=ax.transAxes, va='top', ha='center', fontsize=20)
    else:
        #ax.text(0.65, 0.82, '2.2 g/L PPO + 2.2g/L bis-MSB\nR < 4.0 m', transform=ax.transAxes, va='top', ha='center', fontsize=20)
        ax.text(0.65, 0.69, '2.2 g/L PPO \nR < 4.0 m', transform=ax.transAxes, va='top', ha='center', fontsize=20)
    
    fig.tight_layout()
    fig.savefig("results/plots/iterchi2.png", bbox_inches='tight')
    fig.savefig("results/plots/iterchi2.pdf", bbox_inches='tight')

    fig,ax = plt.subplots(figsize=(10, 6))
    ax.plot(iteration,np.log(train_Y),'o-',color="blue", linewidth=2, markersize=4)
    ax.minorticks_on()
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"log($\chi^2$)", labelpad=5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=3)
    ax.text(0.65, 0.95, 'SNO+ Preliminary', transform=ax.transAxes, va='top', ha='center', fontsize=20)
    if particle_label:
        ax.text(0.65, 0.82, particle_label, transform=ax.transAxes, va='top', ha='center', fontsize=20)
        ax.text(0.65, 0.69, '2.2 g/L PPO \nR < 4.0 m', transform=ax.transAxes, va='top', ha='center', fontsize=20)
        
        #ax.text(0.65, 0.69, '2.2 g/L PPO + 2.2g/L bis-MSB\nR < 4.0 m', transform=ax.transAxes, va='top', ha='center', fontsize=20)
    else:
        #ax.text(0.65, 0.82, '2.2 g/L PPO + 2.2g/L bis-MSB\nR < 4.0 m', transform=ax.transAxes, va='top', ha='center', fontsize=20)
        ax.text(0.65, 0.69, '2.2 g/L PPO \nR < 4.0 m', transform=ax.transAxes, va='top', ha='center', fontsize=20)
    fig.tight_layout()
    fig.savefig("results/plots/iterlogchi2.png", bbox_inches='tight')
    fig.savefig("results/plots/iterlogchi2.pdf", bbox_inches='tight')
    