#!/bin/bash

JOB_ID=$1
MACRO_NAME=$2
RUNID=$3
echo "######### In SIMULATION PROCESS #########" >> algo_log.txt

#source /home/huangp/rat7015/rat/wei_env_Will.sh
source /home/huangp/fresh_rat/rat/wei_env_Will.sh
source /data/snoplus3/weiii/botorch_venv/bin/activate
#rm -rf /data/snoplus2/weiiiii/pytor_bayesian_opti/*.root

rat -n ${RUNID} -N 120 -o "/data/snoplus2/weiiiii/pytor_bayesian_opti/output_${JOB_ID}.ntuple.root" results/macros/${MACRO_NAME}
#rat -n 354109 -N 120 -o "/data/snoplus2/weiiiii/pytor_bayesian_opti/output_${JOB_ID}.root" results/macros/${MACRO_NAME}
#rat -n 354109 -N 500 -o "/data/snoplus2/weiiiii/pytor_bayesian_opti/output_${JOB_ID}.root" results/macros/${MACRO_NAME}

echo "######### END SIMULATION PROCESS #########" >> algo_log.txt

