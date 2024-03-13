source /work/ab0995/eflows4hpc/software/env.sh
module load singularity
export SINGULARITYENV_JAVA_TOOL_OPTIONS=-Xss1280k
export SINGULARITYENV_ENV_LOAD_MODULES_SCRIPT=$PWD/fesom2/env/levante_container_env.sh
