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
cp currentparams_clean.json currentparams.json
cp submit_files/simulate_template.submit submit_files/simulate.submit