import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

if __name__ == "__main__":
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



    fig,ax = plt.subplots()
    ax.plot(iteration,train_Y,'o-',color="blue")
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"$\chi^2$")
    fig.savefig("results/plots/iterchi2.png")
    fig.savefig("results/plots/iterchi2.pdf")

    fig,ax = plt.subplots()
    ax.plot(iteration,np.log(train_Y),'o-',color="blue")
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"log($\chi^2$)")
    fig.savefig("results/plots/iterlogchi2.png")
    fig.savefig("results/plots/iterlogchi2.pdf")
    