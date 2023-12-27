import sys


# COMPSs/PyCOMPSs imports
from pycompss.api.api import compss_wait_on
from pycompss.api.api import TaskGroup
from pycompss.api.api import compss_barrier_group
from pycompss.api.api import compss_cancel_group
from pycompss.api.parameter import *
from pycompss.api.on_failure import on_failure
from pycompss.api.exceptions import COMPSsException
from pycompss.api.mpmd_mpi import mpmd_mpi

# project imports
from esm_ensemble_init import *
from hecuba_lib.esm_dynamic_analysis_results import esm_dynamic_analysis_results


### to_continue in IN mode may be the problem since it behaves as inmutable (not sure...check COMPSs documentation)
# @on_failure(management='IGNORE')
# @mpi(binary="${FESOM_EXE}", runner="mpirun", processes="${FESOM_CORES}", working_dir="{{working_dir_exe}}", fail_by_exit_value=True, processes_per_node=48)
# @task(log_file={Type:FILE_OUT, StdIOStream:STDOUT}, working_dir_exe={Type:INOUT, Prefix:"#"}, to_continue={Type:IN, Prefix:"#"}, returns=int)
# def esm_simulation(log_file, working_dir_exe, to_continue):
#   pass

# original values was mpirun
@on_failure(management='IGNORE')
@mpmd_mpi(runner="srun", working_dir="{{working_dir_exe}}", fail_by_exit_value=True,
          programs=[
              dict(binary="./fesom.x", processes=144, args=""),
              dict(binary="./master.exe", processes=128, args="-v ecmwf -e awi3"),
              dict(binary="./rnfmap.exe", processes=1, args="")
          ])
@task(log_file={Type: FILE_OUT, StdIOStream: STDOUT}, working_dir_exe={Type: INOUT, Prefix: "#"},
      to_continue={Type: IN, Prefix: "#"}, returns=int)
def esm_coupled_simulation(log_file, working_dir_exe, to_continue):
    pass


@on_failure(management='IGNORE')
@task(returns=bool)  # Jorge: Prefix is only needed in @mpi or @binary to avoid to pass the parameter to the binary execution, res={Type:IN, Prefix:"#"})
def esm_member_checkpoint(exp_id, sdate, res):
    # retrieve from Hecuba the last status of the ensemble members produced by the analysis (running in parallel)
    print("Checking status member - " + sdate)
    print("%%%%%%%%%%%%%%%%%% res val is " + str(res))
    ensemble_status = esm_dynamic_analysis_results.get_by_alias(exp_id + "_esm_dynamic_analysis")
    print("%%%%%%%%%%%%%%%%%% status for member " + sdate + " is " + str(ensemble_status))
    to_continue = bool(ensemble_status.results[sdate])
    if not to_continue:
        raise COMPSsException("Member diverged - Aborting member " + str(sdate))
    else:
        return to_continue


@on_failure(management='IGNORE')
@task(returns=bool)
def esm_member_disposal(exp_id, sdate, top_working_dir):
    # TODO: remove hecuba data aswell of the concerned aborted member
    # path
    path = os.path.join("/home/bsc32/bsc32044/results/output_core2/", sdate)
    # removing directory
    shutil.rmtree(path)
    return True


# dummy method to test data exchange with Hecuba
@task(returns=bool)
def esm_dynamic_analysis(exp_id):
    # while True:

    try:

        print("######################## performing dynamic analysis for experiment " + exp_id + "###################")
        # create a dummy object
        # TODO: here is the launching point of the analysis, it will be a PyCOMPSs task
        ds = esm_dynamic_analysis_results()
        ds.results["2000"] = True
        # ds.results["1958"] = False
        # ds.results["1968"] = False
        ds.make_persistent(str(exp_id) + "_esm_dynamic_analysis")


    except COMPSsException:
        pass

    # time.sleep(120)


if __name__ == "__main__":
    print("Running awicm3 coupled - Pycompss")
    # 0 - create a ramdon nr for the experiment id
    exp_id = str(sys.argv[1])

    # create dummy data in Hecuba
    esm_dynamic_analysis(exp_id)

    # prepare folder structure/configuration
    exp_settings = compss_wait_on(esm_ensemble_init(exp_id, True))
    # exp_settings =  compss_wait_on(exp_settings)
    print("##################################### Initialization completed ####################################")

    # sys.exit(0)

    # run the experiment
    ### for sdate in sdates:
    ###     with Task_group(exp_id+"."+sdate, false):
    ###         for time_step in time_steps:
    ###             res= esm_task(...., working_dir)
    ###             esm_checkpoint(..., res)
    ##################################### working code #########################################
    sdates_list = (exp_settings['common']['ensemble_start_dates']).split()
    top_working_dir = exp_settings['common']['top_working_dir']
    to_continue = True
    for sdate in sdates_list:
        # 2 - create a task group for each ESM member and launch all of them in parallel
        ##   working_dir_exe = top_working_dir + "/" + str(exp_id)
        ##   log = working_dir_exe + "/" + "awicm3_2000_1.out"
        ##   res = esm_coupled_simulation(log, working_dir_exe, to_continue)
        ##   print( "################## result " + str(res) + " in " + working_dir_exe + "######################")
        ##############################################################################################

        with TaskGroup(str(exp_id) + "_" + sdate, False):
            # 3 - Launch each SIM, create a implicit dependence by passing the result to the next task (checkpoint)
            n_sims = int(exp_settings['common']['chunks'])
            print("We have " + str(n_sims) + " chunks ")
            to_continue = True
            for sim in range(1, n_sims + 1):
                working_dir_exe = top_working_dir + "/" + str(exp_id) + "/" + str(sdate)
                log = working_dir_exe + "/" + "awicm3_" + str(exp_id) + "_" + str(sdate) + "_" + str(sim) + ".out"
                print("################## launching simulation " + sdate + "." + str(
                    sim) + " in " + working_dir_exe + "######################")
                # res = esm_simulation(log, working_dir_exe, to_continue)
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

    ### for sdate in sdates:
    ###     try:
    ###         compss_barrier_group(exp_id+"."+sdate)
    ###     except COMPSException:
    ###         clean_up(exp_id+"."+sdate)

    for sdate in sdates_list:
        try:
            compss_barrier_group(str(exp_id) + "_" + sdate)
        except COMPSsException:
            # React to the exception (maybe calling other tasks or with other parameters)
            print("ABORTING MEMBER " + sdate)
            # we cancel the whole group
            compss_cancel_group(str(exp_id) + "_" + sdate)
            # clean generated data
            esm_member_disposal(exp_id, sdate, top_working_dir)
            pass

