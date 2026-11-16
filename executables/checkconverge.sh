#!/bin/bash
source /data/snoplus/software/snocave_Alma9/set_env.sh
source /data/snoplus3/weiii/botorch_venv/bin/activate
# Log the current directory and environment variables
echo "Current directory: $(pwd)" >> env.txt
echo "Environment variables: $(env)" >> env.txt
export PYTHONUSERBASE="$HOME/.local"
echo "######### START CONVERGE CHECKS #########" >> algo_log.txt
echo "Current Dir $(pwd)" >> algo_log.txt
python3 src/checkconverge.py >> algo_log.txt 2>&1
status=$?
echo "checkconverge.py exit code: $status" >> algo_log.txt 
echo "######### FINISH CONVERGE CHECKS #########" >> algo_log.txt
# Exit with the status of the Python script
exit $status
