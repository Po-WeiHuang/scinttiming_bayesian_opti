#!/bin/bash

rm -rf *.log
rm -rf algo_log.txt
rm -rf results/pars/*
rm -rf results/plots/*
rm -rf results/macros/*
#rm -rf  dag_running/*
rm -rf results/condor_logs/residuals_errors/*.err
rm -rf results/condor_logs/residuals_outputs/*.out
rm -rf results/condor_logs/residuals_logs/*.log
rm -rf results/condor_logs/select_errors/*.err
rm -rf results/condor_logs/select_outputs/*.out
rm -rf results/condor_logs/select_logs/*.log
rm -rf results/condor_logs/sim_errors/*.err
rm -rf results/condor_logs/sim_outputs/*.out
rm -rf results/condor_logs/sim_logs/*.log
rm -rf dag_running/*
cp opto.dag dag_running/opto.dag
cp training.dag dag_running/training.dag
# optional optics choice: ./clean_files.sh ppo | bismsb  -> activates paramsbound_<phase>.json
if [ -n "$1" ]; then cp "paramsbound_$1.json" paramsbound.json || exit 1; fi
# start point comes from the active config's "StartParams" (default: currentparams_clean.json)
START="$(python3 -c 'import json; print(json.load(open("paramsbound.json")).get("StartParams", "currentparams_clean.json"))')"
cp "$START" currentparams.json || exit 1
echo "Start parameters: $START -> currentparams.json"
echo "To start elsewhere, edit \"StartParams\" in paramsbound.json, then rerun ./clean_files.sh (no argument)."
cp submit_files/simulate_template.submit submit_files/simulate.submit