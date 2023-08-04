module use /apps/HECUBA/modulefiles
module load Hecuba/1.2 # yolanda: update to a new module of hecuba
#module load Hecuba/1.0_api
# Hecuba alreadu loads a special version of Python so not needed at this stage
#module load python
# experiment configuration
# currently hardcoded....check how to set this from alien4cloud

echo "################### preparing env for workflow run ######################"

export FESOM_CORES=144
export QOS=debug
export MEMBERS=1
#export FESOM_WORKINGDIR="/home/bsc32/bsc32044/pycompss_workflow_tests/1948"
export FESOM_EXE="/home/bsc32/bsc32044/fesom_hecuba/bin/fesom.x"
#export NODE_ALLOCATION=$(((${FESOM_CORES}/48)*${MEMBERS}))
export NODE_ALLOCATION=1
echo "Number of cores: ${FESOM_CORES}"
echo "Number of nodes to be used: ${NODE_ALLOCATION}"

#added by support to prevent the segmentation fault
ulimit -Ss unlimited


# to address issue with srun
export COMPSS_MPI_TYPE=impi

# Sample invocation of this script:
# ./launch_mn4.sh 288 debug 1
# ./launch_mn4.sh 144 debug 3
# $1 number of cores needed by the task
# $2 QoS to be used (determines the queue)
# $3 number of simulations of the ensemble

export EXP_ID=$(printf "%06d\n" $((1 + $RANDOM % 100000)))

echo "################### env for workflow run DONE  ######################"

