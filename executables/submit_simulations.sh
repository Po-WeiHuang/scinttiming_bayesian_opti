#!/bin/bash

JOB_ID=$1
MACRO_NAME=$2
RUNID=$3   # from paramsbound.json "RunID", passed by src/update_simulate_submit.py
echo "######### In SIMULATION PROCESS #########" >> algo_log.txt

source /data/snoplus/software/snocave_Alma9/set_env.sh
source /data/snoplus3/weiii/botorch_venv/bin/activate

OUTFILE="/data/snoplus2/weiiiii/pytor_bayesian_opti/output_${JOB_ID}.ntuple.root"
rat -n ${RUNID} -N 120 -o "${OUTFILE}" results/macros/${MACRO_NAME}
RAT_STATUS=$?

# fail the job (so DAGMan retries it) if rat failed or left no output, e.g. a RATDB server error
if [ ${RAT_STATUS} -ne 0 ] || [ ! -s "${OUTFILE}" ]; then
    echo "SIMULATION ${JOB_ID} FAILED: rat exit ${RAT_STATUS}, output ${OUTFILE}" >> algo_log.txt
    exit 1
fi
echo "######### END SIMULATION PROCESS #########" >> algo_log.txt
