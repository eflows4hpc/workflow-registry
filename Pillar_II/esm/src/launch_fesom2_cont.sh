#!/bin/bash

# Code used:
# - https://stackoverflow.com/a/29754866 CC-BY-SA 4.0
# - https://stackoverflow.com/a/246128 CC-BY-SA 4.0
set -o errexit -o pipefail -o noclobber -o nounset

# This script is used to launch experiments with FESOM2.

MODEL="fesom2"

HPC=""
DEBUG=""
PRUNE=""
CORES=0
CORES_PER_NODE=0
START_DATES=""
QOS=""
MEMBERS=1

# Parse options. Note that options may be followed by one colon to indicate
# they have a required argument.
if ! options=$(getopt --name "$(basename "$0")" --options hdc:q: --longoptions help,debug,prune,cores:,start_dates:,members:,qos:,hpc:,cores_per_node: -- "$@"); then
  # Error, getopt will put out a message for us
  exit 1
fi

eval set -- "${options}"

function usage() {
  echo "Usage: $0 --hpc <mn4|levante|local> -c|--cores <CORES> --cores_per_node <CORES_PER_NODE> --start_dates <YYYY,YYYY> --members <MEMBERS> [-q|--qos <QUEUE>] [-d|--debug] [-h|--help]" 1>&2
  exit 1
}

while [ $# -gt 0 ]; do
  # Consume next (1st) argument
  case "$1" in
  -h | --help)
    usage
    ;;
  -d | --debug)
    DEBUG="--debug"
    ;;
  --prune)
    PRUNE="--prune"
    ;;
  # Options with required arguments, an additional shift is required
  -c | --cores)
    CORES="$2"
    shift
    ;;
  --start_dates)
    START_DATES="$2"
    shift
    ;;
  --members)
    MEMBERS="$2"
    shift
    ;;
  -q | --qos)
    QOS="$2"
    shift
    ;;
  --hpc)
    HPC="${2,,}"
    shift
    ;;
  --cores_per_node)
    CORES_PER_NODE="$2"
    shift
    ;;
  --)
    shift
    break
    ;;
  -*)
    echo "$0: error - unrecognized option $1" 1>&2
    usage
    ;;
  *)
    usage
    ;;
  esac
  # Fetch next argument as 1st
  shift
done

if [ -z "${HPC}" ]; then
  echo -e "Please provide a valid HPC environment name\n"
  usage
fi

if [ "${CORES}" -le 0 ]; then
  echo -e "Cores must be equal or greater than 1\n"
  usage
fi

if [ "${CORES_PER_NODE}" -le 0 ]; then
  echo -e "Cores per node must be equal or greater than 1\n"
  usage
fi

if [ -z "${START_DATES}" ]; then
  echo -e "Start dates must be present, with spaces as separators in a single string\n"
  usage
fi

if [ "${MEMBERS}" -le 0 ]; then
  echo -e "Number of ensemble members must be equal or greater than 1\n"
  usage
fi

printf "Launching %s eFlows4HPC ESM experiment...\U1F680\n" "${MODEL}"

# Hecuba configuration
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck source=src/functions.sh
source "${SCRIPT_DIR}/functions.sh"

HECUBA_CONFIGURATION="$(realpath -e -- "${SCRIPT_DIR}/storage_props.cfg")"

NUMBER_OF_START_DATES=$(count_start_dates "${START_DATES}")

check_number_of_cores_node "${CORES}" "${CORES_PER_NODE}"
NODE_ALLOCATION=$(get_nodes_allocated "${CORES}" "${CORES_PER_NODE}" "${NUMBER_OF_START_DATES}")

echo -e "\nLaunch arguments:\n"

# Sample invocation of this script:
#
# ./launch.sh --hpc mn4 --cores 288 --qos debug --start_dates "1990 1991"

# --expid is now optional. Python does the same thing.
EXP_ID=$(get_expid)

echo "EXPID           : ${EXP_ID}"
echo "MODEL           : ${MODEL}"
echo "HPC             : ${HPC}"
echo "CORES           : ${CORES}"
echo "CORES PER NODE  : ${CORES_PER_NODE}"
echo "NODES           : ${NODE_ALLOCATION}"
echo "QOS             : ${QOS}"
echo "START DATES     : (${NUMBER_OF_START_DATES}) ${START_DATES}"
echo "MEMBERS         : ${MEMBERS}"

# NOTE: For the container this may be necessary?
# --env_script="${SCRIPT_DIR}/${MODEL}/env/${HPC}.sh" \

# Use an example so shellcheck can at least check that one when parsing
# this file (you can lint all files independently from this).

# shellcheck source=src/fesom2/env/mn4.sh
source "${SCRIPT_DIR}/${MODEL}/env/${HPC}_container.sh"

CONTAINER=/work/ab0995/eflows4hpc/images/pillar_ii_esm_skylake_openmpi_4_nogpu_v_latest.sif
DATA_PATH=/work/ab0995/eflows4hpc/data/
OUT_PATH=/work/ab0995/eflows4hpc/output/
OUTPUT_DIR=$OUT_PATH/output_dir
mkdir -p $OUTPUT_DIR
TOP_WORKING_DIR=$OUT_PATH/out/top_working_dir
mkdir -p $TOP_WORKING_DIR
# From PyCOMPSs docs: "Expected execution time of the application (in minutes)".
_PYCOMPSS_EXEC_TIME="${_PYCOMPSS_EXEC_TIME:-120}"

# Launch the ESM ensemble simulation with Hecuba infrastructure using COMPSs.
# N.B.: HECUBA_ROOT is defined when you load a Hecuba HPC Module (or manually).

enqueue_compss \
  --debug \
  --project_name=ab0995 \
  --container_image=$CONTAINER \
  --container_opts="-e" \
  --container_compss_path="/opt/view/compss" \
  --env_script="/esm/src/${MODEL}/env/container.sh" \
  --storage_props="${HECUBA_CONFIGURATION}" \
  --storage_home="${HECUBA_ROOT}/compss" \
  --storage_container_image=$CONTAINER \
  --exec_time="${_PYCOMPSS_EXEC_TIME}" \
  --keep_workingdir \
  --worker_working_dir="${SCRIPT_DIR}" \
  --worker_in_master_cpus="${CORES_PER_NODE}" \
  --num_nodes="${NODE_ALLOCATION}" \
  --pythonpath="/esm/src/" \
  --log_dir="${PWD}/out" \
  "/esm/src/esm_simulation.py" \
  --model "${MODEL}" \
  --start_dates "${START_DATES}" \
  --members "${MEMBERS}" \
  --processes "${CORES}" \
  --processes_per_node "${CORES_PER_NODE}" \
  --expid "${EXP_ID}" \
  --config "/esm/src/fesom2/esm_ensemble_tmpl.conf" \
  --mesh_dir "$DATA_PATH/core2/" \
  --data_dir "$DATA_PATH/global/" \
  --top_working_dir "$TOP_WORKING_DIR" --output_dir "$OUTPUT_DIR" \
  "${DEBUG}" \
  "${PRUNE}"
