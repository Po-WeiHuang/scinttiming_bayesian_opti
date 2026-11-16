#!/bin/bash

source /data/snoplus/software/snocave_Alma9/set_env.sh
source /data/snoplus3/weiii/botorch_venv/bin/activate
export PYTHONPATH=/home/huangp/pytor_bayesian_opti:$PYTHONPATH
python3 src/select_parameters.py
