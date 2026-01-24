#!/bin/bash
source ${RATPATH}
source ${VIRTPYTHONPATH}
# Log the current directory and environment variables
echo "Current directory: $$(pwd)" >> env.txt
echo "Environment variables: $$(env)" >> env.txt
export PYTHONUSERBASE="$$HOME/.local"
echo "######### START CONVERGE CHECKS #########" >> algo_log.txt
echo "Current Dir $$(pwd)" >> algo_log.txt
python3 src/checkconverge.py >> algo_log.txt 2>&1
status=$$?
echo "checkconverge.py exit code: $$status" >> algo_log.txt 
echo "######### FINISH CONVERGE CHECKS #########" >> algo_log.txt
# Exit with the status of the Python script
exit $$status
