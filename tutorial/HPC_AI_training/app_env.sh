old_PYTHONPATH=$PYTHONPATH
. /etc/profile.d/z10_spack_environment.sh
export ALYA_BIN=/opt/view/bin/alya
export ALYA_PROCS=28
export ALYA_PPN=112
export ALYA_RUNNER=mpirun
export ComputingUnits=1
export PYTHONPATH=$PYTHONPATH:$old_PYTHONPATH:/usr/local/lib/python3.8/dist-packages/:/usr/lib/python3/dist-packages/
export COMPSS_HOME=/opt/view/compss
export LD_LIBRARY_PATH=$COMPSS_HOME/Bindings/bindings-common/lib/:$LD_LIBRARY_PATH
export COMPSS_MPI_TYPE=ompi
export PATH=/opt/view/bin/:$PATH

