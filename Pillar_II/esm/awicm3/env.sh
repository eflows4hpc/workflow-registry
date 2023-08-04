# Modules needed for executing
module use /apps/HECUBA/modulefiles
module load Hecuba/2.0 # yolanda: update to a new module of hecuba
module load impi/2018.4
module load netcdf/4.4.1.1
module load eccodes/2.8.0
module load hdf5/1.8.19

#module load Hecuba/1.0_api
# Hecuba alreadu loads a special version of Python so not needed at this stage
#module load python
# experiment configuration
# currently hardcoded....check how to set this from alien4cloud

echo "################### preparing env for workflow run ######################"

export QOS=debug
export MEMBERS=3
export USE_HECUBA="TRUE"
# 7 nodes per ensemble  member
export NODE_ALLOCATION=7
[[ "$1" == "TRUE" ]] && USE_HECUBA="TRUE" || USE_HECUBA="FALSE"
echo "Number of cores: ${FESOM_CORES}"
echo "Number of nodes to be used: ${NODE_ALLOCATION}"

#added by support to prevent the segmentation fault
ulimit -Ss unlimited


# to address issue with srun

export OIFS="./master.exe"
export OIFS_CORES=128
export FESOM="./fesom.x"
export FESOM_CORES=144
export RNFMAPPER="./rnfmap.exe"
export RNFMAPPER_CORES=1

export FESOM_USE_CPLNG="active"
export ECE_CPL_NEMO_LIM="false"
export ECE_CPL_FESOM_FESIM="false"
export ECE_AWI_CPL_FESOM="true"

# to address issue with srun
COMPSS_MPI_TYPE=impi
export ECCODES_SAMPLES_PATH=/apps/ECCODES/2.8.0/INTEL/share/eccodes/ifs_samples/grib1/

###################################Prepare rundir#########################################
#export EXP_ID=$(printf "%06d\n" $((1 + $RANDOM % 100000)))
# initially hardcoded, since this should be done in Yorc orchestration, must match the one passed as argument in the HPCWaaS app
export EXP_ID="a0000001"
# prepare work folder
mkdir /home/bsc32/bsc32044/awicm3_ensemble_alien4cloud/$EXP_ID
echo "################### env for workflow run DONE  ######################"
