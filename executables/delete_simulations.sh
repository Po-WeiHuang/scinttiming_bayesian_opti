#!/bin/bash
source /data/snoplus3/weiii/botorch_venv/bin/activate
SIMDIR="/data/snoplus2/weiiiii/pytor_bayesian_opti"
# ${SIMDIR:?} stops the script instead of deleting from / if the directory is empty
rm -f "${SIMDIR:?}"/output_*.ntuple.root

echo "######### DELETE SIMULATION FILES #########" >> algo_log.txt
