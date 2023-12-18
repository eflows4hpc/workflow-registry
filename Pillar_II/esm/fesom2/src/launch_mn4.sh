#todo: to be remplaced by python file
#todo: add also the launching of Hecuba storage, is it possible based on this hint
#https://compss-doc.readthedocs.io/en/stable/Sections/06_Persistent_Storage/05_Own_interface.html?highlight=enqueue_compss#using-enqueue-compss

#module load COMPSs
# COMPSs  version recommended for this project is the eflows4hpc
module load COMPSs/3.2
module use /apps/HECUBA/modulefiles
module load Hecuba/2.1_intel # yolanda: update to a new module of hecuba
module load singularity

#module load Hecuba/1.0_api
# Hecuba alreadu loads a special version of Python so not needed at this stage
#module load python
# experiment configuration
# only parameter - #CORES
export FESOM_CORES=$1
export QOS=$2
export MEMBERS=$3
#export FESOM_WORKINGDIR="/home/bsc32/bsc32044/pycompss_workflow_tests/1948"
export FESOM_EXE="/gpfs/projects/dese28/models/fesom2_eflows4hpc/fesom2/bin/fesom.x"
#export NODE_ALLOCATION=$(((${FESOM_CORES}/48)*${MEMBERS}))
export NODE_ALLOCATION=$(((${FESOM_CORES}/48)*${MEMBERS}))
echo "Number of cores: ${FESOM_CORES}"
echo "Number of nodes to be used: ${NODE_ALLOCATION}"

#added by support to prevent the segmentation fault
ulimit -Ss unlimited


# to address issue with srun
COMPSS_MPI_TYPE=impi

# Sample invocation of this script:
# ./launch_mn4.sh 288 debug 1
# ./launch_mn4.sh 144 debug 3
# $1 number of cores needed by the task
# $2 QoS to be used (determines the queue)
# $3 number of simulations of the ensemble

EXP_ID=$(printf "%06d\n" $((1 + $RANDOM % 100000)))


# launch the esm ensemble simulation with hecuba infraestructure through COMPSs (WORKING)
enqueue_compss -t -g -d --sc_cfg=mn.cfg  \
               --qos=${QOS}  \
               --storage_props=/gpfs/projects/dese28/eflows4hpc/esm/fesom2/src/hecuba_lib/storage_props.cfg \
               --storage_home=$HECUBA_ROOT/compss \
               --job_name=esm_workflow  \
               --exec_time=120  \
               --keep_workingdir \
               --worker_working_dir=$PWD \
               --worker_in_master_cpus=48  \
               --num_nodes=${NODE_ALLOCATION}  \
               --pythonpath=$PWD:$HECUBA_ROOT/compss esm_simulation.py ${EXP_ID}


