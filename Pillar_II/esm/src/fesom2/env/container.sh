old_PYTHONPATH=$PYTHONPATH
. /etc/profile.d/z10_spack_environment.sh
export PYTHONPATH=/opt/view/lib/python3.8/site-packages/Hecuba-2.1.3-py3.8-linux-x86_64.egg/:/opt/view/lib/python3.8/site-packages/cassandra_driver-3.28.0-py3.8-linux-x86_64.egg/:/opt/view/lib/python3.8/site-packages/geomet-0.2.1.post1-py3.8.egg/:/opt/view/lib/python3.8/site-packages/click-8.1.7-py3.8.egg/:$PYTHONPATH:$old_PYTHONPATH
export COMPSS_HOME=/opt/view/compss
export LD_LIBRARY_PATH=$COMPSS_HOME/Bindings/bindings-common/lib/:$LD_LIBRARY_PATH
export MPI_RUNNER_SCRIPT="/gpfs/projects/bsc19/eflows_demo/pillarII/hecuba_test/fesom2/mpi_run.sh"
export SINGULARITYENV_MPI_RUNNER_SCRIPT=$MPI_RUNNER_SCRIPT
#added by support to prevent the segmentation fault
ulimit -Ss unlimited
# to address issue with srun
export COMPSS_MPI_TYPE=ompi
export SINGULARITYENV_CONTACT_NAMES=$CONTACT_NAMES

