#!/bin/bash
#source /home/huangp/rat7015/rat/wei_env_Will.sh
source /home/huangp/fresh_rat/rat/wei_env_Will.sh
source /data/snoplus3/weiii/botorch_venv/bin/activate
export PYTHONPATH=/home/huangp/pytor_bayesian_opti:$PYTHONPATH
#cd /home/huangp/bayesian_optimisation
python3 src/time_residuals.py