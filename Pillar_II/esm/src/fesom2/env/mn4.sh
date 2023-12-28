#!/bin/bash

# Environment settings for FESOM2 on MN4.

# HPC modules.
module load COMPSs/3.2
module use /apps/HECUBA/modulefiles
module load Hecuba/2.1_intel
module load singularity

# Added by support to prevent the segmentation fault.
ulimit -Ss unlimited

# To address issue with srun.
export COMPSS_MPI_TYPE=impi
