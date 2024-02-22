#!/bin/bash

# Common functions for the ESM shell code.

set -o errexit -o pipefail -o noclobber -o nounset

# Count the number of start dates. The start dates are given
# as a single string, where each date is separated by commas.
# It ignores spaces, trailing and leading commas and spaces.
#
# Globals:
#   None.
# Arguments:
#   String with start dates and commas.
# Outputs:
#   The number of start dates.
function count_start_dates() {
  if [ -z "$1" ]; then
    echo "You must specify a valid value for start dates. Given: [$1]" >&2
    exit 1
  fi

  local ARR_STRING="$1"
  # Replaces spaces (the / /) everywhere (the double // does that)
  # by nothing (the empty value after the last /).
  ARR_STRING="${ARR_STRING// /}"

  # From: https://stackoverflow.com/a/61087835
  # Enable pathname expansion.
  shopt -s extglob
  # Remove leading commas, then spaces and tabs.
  ARR_STRING=("${ARR_STRING[@]/#+([,])/}")
  ARR_STRING=("${ARR_STRING[@]/#+([[:blank:]])/}")
  # Remove trailing commas, then spaces and tabs.
  ARR_STRING=("${ARR_STRING[@]/%+([,])/}")
  ARR_STRING=("${ARR_STRING[@]/%+([[:blank:]])/}")
  shopt -u extglob

  oldIFS=$IFS
  IFS=','
  local START_DATES_ARRAY=
  read -r -a START_DATES_ARRAY <<<"${ARR_STRING[@]}"
  local NUMBER_OF_START_DATES="${#START_DATES_ARRAY[@]}"
  IFS=$oldIFS
  echo "${NUMBER_OF_START_DATES}"
}

# Check and warn the user whenever we underuser a node.
# It prints warning messages to the stdout depending on
# the number of cores used and the number of cores per
# node.
#
# Globals:
#   None.
# Arguments:
#   Number of cores.
#   Number of cores per node.
# Outputs:
#   None.
function check_number_of_cores_node() {
  local CORES="$1"
  local CORES_PER_NODE="$2"
  if [ "${CORES}" -lt "${CORES_PER_NODE}" ]; then
    echo "WARNING: You have ${CORES_PER_NODE} cores per node, but requested less: ${CORES}"
  elif [ $((CORES % CORES_PER_NODE)) -ne 0 ]; then
    echo "WARNING: You are not using all the cores of your nodes (${CORES_PER_NODE}), you requested: ${CORES}"
  fi
}

# Get the node allocation. This is calculated based on the number of
# cores per node.
# - 48 cores per node, you request 1 core: 1 node allocated
# - 48 cores per node, you request 48 cores: 1 node allocated
# - 48 cores per node, you request 49 nodes: 2 nodes allocated
# - 48 cores per node, you request 96: 2 nodes allocated
# - 48 cores per node, you request 100: 3 nodes allocated
# Invalid values raise an error.
#
# Globals:
#   None.
# Arguments:
#   Number of cores.
#   Number of cores per node.
#   Number of start dates.
# Outputs:
#   The total number of nodes allocated.
function get_nodes_allocated() {
  local CORES="$1"
  local CORES_PER_NODE="$2"
  local NUMBER_OF_START_DATES="$3"

  # If FESOM2 needs 2 cores, and we have 2 start dates, then
  # we will want to have at least 4 cores for FESOM2. Hence:
  local TOTAL_CORES_NEEDED="$((NUMBER_OF_START_DATES * CORES))"

  # N.B.: No math module, nor ceil in Bash, hence this
  #       confusing code to replicate this:
  #       `math.ceil(needed cores / cores_per_node)`
  NODE_ALLOCATION="$(((TOTAL_CORES_NEEDED + CORES_PER_NODE - 1) / CORES_PER_NODE))"
  echo "${NODE_ALLOCATION}"
}

# Get a new, 6-length, alpha-numeric, experiment ID.
# Starts with a character, followed by 5 digits.
# Follows no special order when creating ID's.
# For example: a12345, i88766, b00022.
function get_expid() {
  ALPHABET="abcdefghijklmnopqrstuvwxyz"
  EXPID_ALPHA="${ALPHABET:$(( RANDOM % ${#ALPHABET} )):1}"
  EXPID_NUMBER="$(shuf -i 10000-99999 -n 1)"
  echo "${EXPID_ALPHA}${EXPID_NUMBER}"
}
