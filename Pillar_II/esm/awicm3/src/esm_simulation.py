import os
import shutil
import sys
from contextlib import suppress
from typing import Any

from pycompss.api.api import TaskGroup  # type: ignore
from pycompss.api.api import compss_barrier_group
from pycompss.api.api import compss_cancel_group
from pycompss.api.api import compss_wait_on
from pycompss.api.exceptions import COMPSsException  # type: ignore
from pycompss.api.mpmd_mpi import mpmd_mpi  # type: ignore
from pycompss.api.on_failure import on_failure  # type: ignore
from pycompss.api.parameter import IN, Type, FILE_OUT, StdIOStream, STDOUT, INOUT, Prefix  # type: ignore
from pycompss.api.task import task  # type: ignore

# project imports
from esm_ensemble_init import esm_ensemble_init  # type: ignore
from hecuba_lib.esm_dynamic_analysis_results import esm_dynamic_analysis_results


@on_failure(management='IGNORE')
@mpmd_mpi(runner="srun", working_dir="{{working_dir_exe}}", fail_by_exit_value=True,
          programs=[
              dict(binary="./fesom.x", processes=144, args=""),
              dict(binary="./master.exe", processes=128, args="-v ecmwf -e awi3"),
              dict(binary="./rnfmap.exe", processes=1, args="")
          ])
@task(log_file={Type: FILE_OUT, StdIOStream: STDOUT}, working_dir_exe={Type: INOUT, Prefix: "#"},
      to_continue={Type: IN, Prefix: "#"}, returns=int)
def esm_coupled_simulation(log_file: str, working_dir_exe: str, to_continue: bool) -> Any:
    return None


@on_failure(management='IGNORE')
# Jorge: Prefix is only needed in @mpi or @binary to avoid to pass the parameter to the binary execution, res={Type:IN, Prefix:"#"})
@task(returns=bool)
def esm_member_checkpoint(exp_id: str, sdate: str, res: Any) -> bool:
    # retrieve from Hecuba the last status of the ensemble members produced by the analysis (running in parallel)
    print("Checking status member - " + sdate)
    print("%%%%%%%%%%%%%%%%%% res val is " + str(res))
    ensemble_status = esm_dynamic_analysis_results.get_by_alias(exp_id + "_esm_dynamic_analysis")
    print("%%%%%%%%%%%%%%%%%% status for member " + sdate + " is " + str(ensemble_status))
    to_continue = bool(ensemble_status.results[sdate])
    if not to_continue:
        raise COMPSsException("Member diverged - Aborting member " + sdate)
    return to_continue


@on_failure(management='IGNORE')
@task(returns=bool)
def esm_member_disposal(exp_id: str, sdate: str, top_working_dir: str) -> bool:
    # TODO: remove hecuba data aswell of the concerned aborted member
    path = os.path.join("/home/bsc32/bsc32044/results/output_core2/", sdate)
    shutil.rmtree(path)
    return True


# dummy method to test data exchange with Hecuba
@task(returns=bool)
def esm_dynamic_analysis(exp_id: str) -> None:
    with suppress(COMPSsException):
        print("######################## performing dynamic analysis for experiment " + exp_id + "###################")
        # TODO: here is the launching point of the analysis, it will be a PyCOMPSs task
        ds = esm_dynamic_analysis_results()
        ds.results["2000"] = True
        # ds.results["1958"] = False
        # ds.results["1968"] = False
        ds.make_persistent(exp_id + "_esm_dynamic_analysis")


def main() -> None:
    print("Running awicm3 coupled - Pycompss")
    exp_id = str(sys.argv[1])

    esm_dynamic_analysis(exp_id)

    exp_settings = compss_wait_on(esm_ensemble_init(exp_id))
    print("##################################### Initialization completed ####################################")

    sdates_list = (exp_settings['common']['ensemble_start_dates']).split()
    top_working_dir = exp_settings['common']['top_working_dir']
    for sdate in sdates_list:
        # Create a task group for each ESM member and launch all of them in parallel
        with TaskGroup(exp_id + "_" + sdate, False):
            # 3 - Launch each SIM, create an implicit dependence by passing the result to the next task (checkpoint)
            n_sims = int(exp_settings['common']['chunks'])
            print("We have " + str(n_sims) + " chunks ")
            to_continue = True
            for sim in range(1, n_sims + 1):
                working_dir_exe = top_working_dir + "/" + exp_id + "/" + sdate
                log = working_dir_exe + "/" + "awicm3_" + exp_id + "_" + sdate + "_" + str(sim) + ".out"
                print("################## launching simulation " + sdate + "." + str(
                    sim) + " in " + working_dir_exe + "######################")
                res = esm_coupled_simulation(log, working_dir_exe, to_continue)
                # check the state of the member, for the first one there is nothing to check
                if sim > 1:
                    print("Checkpoint nr " + str(res))
                    ### the barrier was added just to check if it solves the pruning issue
                    ### adding a barrier causes all tasks to run in serial even if these are in different task groups
                    # compss_wait_on(esm_member_checkpoint(exp_id, sdate, res))
                    to_continue = esm_member_checkpoint(exp_id, sdate, res)
                else:
                    to_continue = res

    for sdate in sdates_list:
        try:
            compss_barrier_group(exp_id + "_" + sdate)
        except COMPSsException:
            # React to the exception (maybe calling other tasks or with other parameters)
            print("ABORTING MEMBER " + sdate)
            # we cancel the whole group
            compss_cancel_group(exp_id + "_" + sdate)
            # clean generated data
            esm_member_disposal(exp_id, sdate, top_working_dir)


if __name__ == "__main__":
    main()
