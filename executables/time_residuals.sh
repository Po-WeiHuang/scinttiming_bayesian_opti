#!/bin/bash
source /data/snoplus/software/snocave_Alma9/set_env.sh
source /data/snoplus3/weiii/botorch_venv/bin/activate
export PYTHONPATH=/home/huangp/pytor_bayesian_opti:$PYTHONPATH
export SIMSAVEPATH=/data/snoplus2/weiiiii/pytor_bayesian_opti   # src/time_residuals.py reads the simulated ntuples from here
python3 src/time_residuals.py
