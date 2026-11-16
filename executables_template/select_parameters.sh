#!/bin/bash

source ${RATPATH}
source ${VIRTPYTHONPATH}
export PYTHONPATH=${WORKINGDIR}:$$PYTHONPATH
python3 src/select_parameters.py
