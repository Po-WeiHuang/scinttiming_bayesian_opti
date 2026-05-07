import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import sys
import os
sys.path.insert(0, "/home/huangp/pytor_bayesian_opti")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PlottingStyle'))
from SNOplus_PythonPublicationStyle import SNOplus_style 
if __name__ == "__main__":
    SNOplus_style()
    with open("currentparams.json", "r") as f:
        params = json.load(f)
    finaliteration = int(params["iter"])
    iteration = np.array([i for i in range(1,finaliteration+1)])
    # Loading all .npy
    dir_path = Path("results/pars")
    arrays = [np.load(p) for p in dir_path.glob(f"*.npy")]
    # Split to parameter_set & objective
    train = np.vstack(arrays)
    train_Y = train[:,-2]



    fig,ax = plt.subplots(figsize=(10, 6))
    ax.plot(iteration,train_Y,'o-',color="blue", linewidth=2, markersize=4)
    ax.minorticks_on()
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"$\chi^2$", labelpad=0)
    fig.tight_layout()
    fig.savefig("results/plots/iterchi2.png", bbox_inches='tight')
    fig.savefig("results/plots/iterchi2.pdf", bbox_inches='tight')

    fig,ax = plt.subplots(figsize=(10, 6))
    ax.plot(iteration,np.log(train_Y),'o-',color="blue", linewidth=2, markersize=4)
    ax.minorticks_on()
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"log($\chi^2$)", labelpad=0)
    fig.tight_layout()
    fig.savefig("results/plots/iterlogchi2.png", bbox_inches='tight')
    fig.savefig("results/plots/iterlogchi2.pdf", bbox_inches='tight')
    