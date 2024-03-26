old_PYTHONPATH=$PYTHONPATH
. /etc/profile.d/z10_spack_environment.sh
export ALYA_BIN=/opt/view/bin/alya
export ALYA_PROCS=96
export ALYA_PPN=48
export ComputingUnits=1
export PYTHONPATH=$PYTHONPATH:$old_PYTHONPATH:/usr/local/lib/python3.8/dist-packages/:/usr/lib/python3/dist-packages/
export COMPSS_HOME=/opt/view/compss
export LD_LIBRARY_PATH=$COMPSS_HOME/Bindings/bindings-common/lib/:$LD_LIBRARY_PATH
export COMPSS_MPI_TYPE=ompi
export MPI_RUNNER_SCRIPT=$PWD/mpi_run.sh
export SINGULARITYENV_MPI_RUNNER_SCRIPT=$MPI_RUNNER_SCRIPT
export PATH=/opt/view/bin/:$PATH
