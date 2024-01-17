# Tests for functions.sh

## setup

setup() {
  bats_load_library bats-support
  bats_load_library bats-assert

  # get the containing directory of this file
  # use $BATS_TEST_FILENAME instead of ${BASH_SOURCE[0]} or $0,
  # as those will point to the bats executable's location or the preprocessed file respectively
  DIR="$( cd "$( dirname "$BATS_TEST_FILENAME" )" >/dev/null 2>&1 && pwd )"
  # source file under test
  source "${DIR}/../src/functions.sh"
}

## count_start_dates

@test "count_start_dates counts dates correctly for a single date" {
  count=$(count_start_dates "1948")
  assert_equal "1" "${count}"
}

@test "count_start_dates counts dates correctly for multiple dates" {
  count=$(count_start_dates "1948,1999,2000")
  assert_equal "3" "${count}"
}

@test "count_start_dates works if spaces are used too" {
  count=$(count_start_dates "1948, 1999, 1000")
  assert_equal "3" "${count}"
}

@test "count_start_dates works with trailing commas" {
  count=$(count_start_dates "1948,")
  assert_equal "1" "${count}"
}

@test "count_start_dates works with trailing spaces" {
  count=$(count_start_dates "1948 ")
  assert_equal "1" "${count}"
}

@test "count_start_dates works with trailing commas and spaces" {
  count=$(count_start_dates "1948, ")
  assert_equal "1" "${count}"
}

@test "count_start_dates works with leading commas" {
  count=$(count_start_dates ",1948")
  assert_equal "1" "${count}"
}

@test "count_start_dates works with multiple leading commas" {
  count=$(count_start_dates ",,,1948")
  assert_equal "1" "${count}"
}

@test "count_start_dates works with leading spaces" {
  count=$(count_start_dates "  1948")
  assert_equal "1" "${count}"
}

@test "count_start_dates works with leading commas and spaces" {
  count=$(count_start_dates "  ,  1948")
  assert_equal "1" "${count}"
}

@test "count_start_dates fails if no values provided" {
  run count_start_dates ""
  assert_output --partial "You must specify a valid value for start dates"
}

## check_number_of_cores_node

@test "check_number_of_cores_node does nothing with valid values" {
  run check_number_of_cores_node "48" "48"
  assert_output ""
}

@test "check_number_of_cores_node does nothing with valid values (more cores)" {
  run check_number_of_cores_node "96" "48"
  assert_output ""
}

@test "check_number_of_cores_node warns if underusing a node" {
  run check_number_of_cores_node "1" "48"
  assert_output --partial "but requested less"
}

@test "check_number_of_cores_node warns if not using all node" {
  run check_number_of_cores_node "50" "48"
  assert_output --partial "You are not using all the cores of your nodes"
}

## get_nodes_allocated

@test "get_nodes_allocated returns the correct number of node (1)" {
  run get_nodes_allocated "1" "48" "1"
  assert_output "1"
}

@test "get_nodes_allocated returns the correct number of node (full node)" {
  run get_nodes_allocated "48" "48" "1"
  assert_output "1"
}

@test "get_nodes_allocated returns the correct number of nodes (2 dates, 1 core for FESOM2)" {
  run get_nodes_allocated "1" "48" "2"
  assert_output "1"
}

@test "get_nodes_allocated returns the correct number of nodes (2 dates, 24 cores for FESOM2)" {
  run get_nodes_allocated "24" "48" "2"
  assert_output "1"
}

@test "get_nodes_allocated returns the correct number of nodes (2 dates, 48 cores for FESOM2)" {
  run get_nodes_allocated "48" "48" "2"
  assert_output "2"
}

@test "get_nodes_allocated returns the correct number of nodes (4 dates, 144 cores for FESOM2)" {
  run get_nodes_allocated "144" "48" "4"
  assert_output "12"
}
