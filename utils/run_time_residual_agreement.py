import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from plot_tres import time_residual_agreement

# Synthetic data: mixture to mimic time residuals
rng = np.random.default_rng(12345)
# main peak near 0 with tails
data = np.concatenate([rng.normal(loc=0, scale=2.0, size=5000), rng.exponential(scale=20.0, size=500)])
# model: slightly shifted and narrower
model = np.concatenate([rng.normal(loc=0.5, scale=1.8, size=4800), rng.exponential(scale=18.0, size=700)])
pars = np.linspace(0.1, 1.0, 10)

# use iter 999 to avoid overwriting
time_residual_agreement(data, model, 999, pars)
print('Done. Plots saved to results/plots/999/')
