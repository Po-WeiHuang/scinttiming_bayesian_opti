#!/bin/bash
# DAGMan allows one PRE script per node: delete the previous iteration's ntuples, then write the
# next macro and simulate.submit. Any failure fails the PRE script, so the node is retried.
set -e
executables/delete_simulations.sh
python3 src/update_simulate_submit.py
