# Using Botorch to perform bayesian optimisation process
# Output a set of parameter to get simulated
from pathlib import Path
import numpy as np
import json
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
import sys
sys.path.insert(0, "/home/huangp/pytor_bayesian_opti")
from utils.plot_surr_acq import surrogate_uncert_acquistion

def get_mean_sigma_EI_frombestAcq_2Dprojection(bounds,gp,candidate,logEI,proj_dim=(0,1)):
    """
    bounds are a (2,D) tensors to define the range in each parameter space
    gp is the gaussian process which contains further posterior distribution
    candidate is a next proposed point (D,)
    proj_dim is projection dimension (tuple-like)
    return: mean(M,N) ,sigma(M,N), EI(M,N), xx(meshed projection space), yy(meshed projection space)
    """
    ref = candidate
    #print("candidate ",candidate)
    dx = proj_dim[0]; dy = proj_dim[1]
    N  = 50; M = 50 # return shape
    x = np.linspace(bounds[0,dx],bounds[1,dx],N) 
    y = np.linspace(bounds[0,dy],bounds[1,dy],M)

    xx,yy = np.meshgrid(x,y)
    grid = np.tile( ref, (N*M,1) ) # (N*M,D)
    grid[:,dx] = xx.flatten()
    grid[:,dy] = yy.flatten()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    grid_t = torch.from_numpy(grid).to(device, dtype=torch.double)

    means_list = []; vars_list = []; acq_list = []
    with torch.no_grad():
        for g in grid_t:
            #print(g)
            post = gp.posterior(g)
            m = post.mean.item()
            v = post.variance.item()
            means_list.append(-m) # recover the negative sign assigned in chi2 during gp process
            #print(f"m:{m}")

            vars_list.append(v)
            av = logEI(g.unsqueeze(0).unsqueeze(1)).item()
            acq_list.append(av)
            #print(f"m:{m} v:{v} av:{av}")
    means = np.array(means_list).reshape(M,N)
    sigma = np.sqrt(np.maximum(np.array(vars_list).reshape(M,N), 0))
    acq   =  np.array(acq_list).reshape(M,N)
    """
    # Testing the parameters from slice actually cover the global min/max
    m_irow = np.where(means==np.min(means))[0][0] 
    m_icol = np.where(means==np.min(means))[1][0] 
    s_irow = np.where(sigma==np.min(sigma))[0][0] 
    s_icol = np.where(sigma==np.min(sigma))[1][0] 
    a_irow = np.where(acq==np.max(acq))[0][0]
    a_icol = np.where(acq==np.max(acq))[1][0]
    print(f"Minimum mean occurs at {means[m_irow,m_icol]}") 
    print(f"Minimum sigma occurs at {sigma[s_irow,s_icol]}") 
    print(f"Minimum acq occurs at {acq[a_irow,a_icol]}") 
    print(f"Next sample points are around {proj_dim[0]}:{xx[a_irow,a_icol]},{proj_dim[1]}:{yy[a_irow,a_icol]}") 
    """
    return means, sigma, acq, xx, yy


if __name__ == "__main__":
    with open("algo_log.txt", "a") as outlog:
        outlog.write("\n\n")
        outlog.write("### THIS IS SELECT_PARAMETERS SCRIPT ###\n")

        # Loading all .npy
        dir_path = Path("results/pars")
        arrays = [np.load(p) for p in dir_path.glob(f"*.npy")]
        # Split to parameter_set & objective
        train = np.vstack(arrays)
        train_X = train[:,:-3]
        train_Y = train[:,-2]
        # accessing bounds
        with open("paramsbound.json", "r") as f:
            params = json.load(f)
        
        bounds = [(params[ikey]) for ikey in params if ikey!="Type"]
        print(bounds)
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        outlog.write(f"Using device {device}")
        train_X_dev = torch.from_numpy(train_X).to(device)
        train_Y_dev = torch.from_numpy(train_Y).to(device).unsqueeze(-1)
        print(train_X_dev.shape)
        print(train_Y_dev.shape)
        train_Y_Bo = -train_Y_dev
        bounds = torch.tensor(bounds).T #from (9,2) to (2,9)
        print("bounds shape", bounds.shape) 
        
        
        gp = SingleTaskGP(
            train_X=train_X_dev,
            train_Y=train_Y_Bo,
            input_transform=Normalize(d=train_X_dev.shape[1], bounds=bounds),
            outcome_transform=Standardize(m=1) if train_Y_Bo.shape[0] > 1 else None,
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        
        logEI = LogExpectedImprovement(model=gp, best_f=train_Y_Bo.max())

        # add amplitude constraint: A1+A2+A3+A4 = 1
        amp_indices = torch.tensor([5,6,7,8], dtype=torch.long, device=device)
        amp_coefs = torch.tensor([1.0,1.0,1.0,1.0], dtype=torch.float32, device=device)

        # equality_constraints expects a list of tuples (indices, coefficients, rhs)
        eq_constraints = [(amp_indices, amp_coefs, 1.0)] 

        candidate, acq_value = optimize_acqf(
        logEI, bounds=bounds, q=1, num_restarts=25, raw_samples=2048, equality_constraints=eq_constraints,
        )
        candidate = candidate.detach().cpu().numpy()
        print(candidate)  
        outlog.write(f"\n Next sample at {candidate}")
        
        with open("currentparams.json", "r") as f:
            curr_params = json.load(f)
            iteration = curr_params["iter"]

        new_params = {
            "T1": float(candidate[0][0]),
            "T2": float(candidate[0][1]),
            "T3": float(candidate[0][2]),
            "T4": float(candidate[0][3]),
            "TR": float(candidate[0][4]),
            "A1": float(candidate[0][5]),
            "A2": float(candidate[0][6]),
            "A3": float(candidate[0][7]),
            "A4": float(candidate[0][8]),
        }

        merged_params = curr_params | new_params

        with open("currentparams.json", "w") as f:
            json.dump(merged_params, f, indent=2)
        if iteration > 1: # only plots surrogate/acquisition when measuing more than 1 point
            subspace = [(0,1),(2,3),(0,4),(5,6),(7,8),(0,5),(1,6),(2,7),(3,8)]
            for projection_dim in subspace:
                mean, sigma, acq, xx, yy = get_mean_sigma_EI_frombestAcq_2Dprojection(bounds,gp,candidate,logEI,proj_dim=projection_dim)
                measured_points_subspace = np.column_stack((train_X[:,projection_dim[0]],train_X[:,projection_dim[1]]))
                surrogate_uncert_acquistion(mean, sigma, acq, xx, yy, projection_dim,iteration,measured_points_subspace)

        outlog.write("\n### END OF SELECT_PARAMETERS SCRIPT ###\n")
        