#!/usr/bin/env bash

export COMPSS_PYTHON_VERSION=3-ML
module use /apps/modules/modulefiles/tools/COMPSs/.custom
module load COMPSs/3.2
module load BackTrackBB_FINAL

echo "Compiling and installing in this folder..."

#python3 -m pip install -e .
export PYTHONPATH="${HOME}/.local/lib/python3.6/site-packages:${PYTHONPATH}"
${HOME}/.local/bin/pip3.6 install -e .

echo "Finished"