#!/bin/bash
source ${VIRTPYTHONPATH}
SIMDIR="${SIMSAVEPATH}"
# $${SIMDIR:?} stops the script instead of deleting from / if the directory is empty
rm -f "$${SIMDIR:?}"/output_*.ntuple.root

echo "######### DELETE SIMULATION FILES #########" >> algo_log.txt
