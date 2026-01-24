#!/bin/bash

JOB_ID=$$1
MACRO_NAME=$$2
echo "######### In SIMULATION PROCESS #########" >> algo_log.txt

source ${RATPATH}
source ${VIRTPYTHONPATH}
#rm -rf /data/snoplus2/weiiiii/pytor_bayesian_opti/*.root

rat -n ${RUNID} -N 50 -o "${SIMSAVEPATH}/output_$${JOB_ID}.root" results/macros/$${MACRO_NAME}

echo "######### END SIMULATION PROCESS #########" >> algo_log.txt

