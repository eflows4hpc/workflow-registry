#!/bin/bash

# Environment settings for FESOM2 on MN4.

## HPC modules.

module load COMPSs/3.2
module use /apps/HECUBA/modulefiles
module load intel/2020.1 impi/2018.4 mkl/2020.1 bsc/1.0 hdf5/1.14.0-gcc pnetcdf/1.12.3-gcc netcdf/c-4.9.2_fortran-4.6.0_cxx4-4.3.1_hdf5-1.14.0_pnetcdf-1.12.3-gcc Hecuba/2.1
export FC=mpif90 CC=mpicc CXX=mpicxx
export NETCDF_Fortran_INCLUDE_DIRECTORIES=/apps/NETCDF/c-4.9.2_fortran-4.6.0_cxx4-4.3.1_hdf5-1.14.0_pnetcdf-1.12.3/GCC/IMPI/include
export NETCDF_C_INCLUDE_DIRECTORIES=/apps/NETCDF/c-4.9.2_fortran-4.6.0_cxx4-4.3.1_hdf5-1.14.0_pnetcdf-1.12.3/GCC/IMPI/include
export NETCDF_CXX_INCLUDE_DIRECTORIES=/apps/NETCDF/c-4.9.2_fortran-4.6.0_cxx4-4.3.1_hdf5-1.14.0_pnetcdf-1.12.3/GCC/IMPI/include
module load singularity

## IT

# Added by support to prevent the segmentation fault.
ulimit -Ss unlimited

## PyCOMPSs

# To address issue with srun.
export COMPSS_MPI_TYPE=impi
# This was hard-coded before in the launch_fesom2.sh script.
# "Expected execution time of the application (in minutes)".
export _PYCOMPSS_EXEC_TIME=120
