#!/bin/bash
source ${RATPATH}
source ${VIRTPYTHONPATH}
export PYTHONPATH=${WORKINGDIR}:$$PYTHONPATH
export SIMSAVEPATH=${SIMSAVEPATH}   # src/time_residuals.py reads the simulated ntuples from here
python3 src/time_residuals.py
