module load COMPSs/3.2
module use /apps/HECUBA/modulefiles
module load Hecuba/2.1
module load singularity
export SINGULARITYENV_JAVA_TOOL_OPTIONS=-Xss1280k
export MPI_RUNNER_SCRIPT="/gpfs/projects/bsc19/eflows_demo/pillarII/hecuba_test/fesom2/mpi_run.sh"
export SINGULARITYENV_MPI_RUNNER_SCRIPT=$MPI_RUNNER_SCRIPT
export SINGULARITYENV_ENV_LOAD_MODULES_SCRIPT=$PWD/fesom2/env/mn4_container_env.sh
