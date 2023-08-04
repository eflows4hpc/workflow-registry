import sys
import os

install_dir=os.environ.get("PTF_INSTALL_DIR")
HySEA_dir=os.environ.get("PTF_HYSEA_DIR") 

sys.path.append(install_dir+'/Code')
sys.path.append(install_dir+'/Code/Common/py/')

from pycompss.api.task import task
from pycompss.api.mpi import mpi
from pycompss.api.binary import binary
from pycompss.api.constraint import constraint
from pycompss.api.parameter import *
from pycompss.api.api import compss_wait_on_file
from pycompss.api.api import compss_barrier
from pycompss.api.api import compss_wait_on

from run_step1 import run_step1_init
from run_kagan import run_step_kagan
from run_mare import run_step_mare
from ptf_parser import parse_ptf_stdin
from Step2_create_ts_input_for_ptf_mod import step2_create_ptf_input
from Step2_extract_ts_mod import step2_extract_ts
from run_step3 import run_step3_init


# Required paths
tsunamiHySEA_bin= HySEA_dir + "/bin/TsunamiHySEA"
load_balancing_bin= HySEA_dir + "/bin/get_load_balancing"

simulBS_bin = install_dir+"/Code/scripts/Step2_config_simul.sh"
config_bin = install_dir+"/Code/scripts/Create_config.sh"

# GPUS size ! gpus_per_node > gpus_per_exec
gpus_per_node=os.environ.get("PTF_GPUS_NODE",4)
gpus_per_exec=os.environ.get("PTF_GPUS_EXEC",4)

###################### PYCOMPSS TASKS #########################

### TASK FOR STEP1 ###
@task(config_file=FILE_IN, returns=1)
def step1_func(args, config_file, seistype, sim_files_step1):
    args.cfg=config_file
    run_step1_init(args,sim_files_step1)
    return sim_files_step1 + "/Step1_scenario_list_"+seistype+".txt"

@binary(binary=config_bin)
@task(config_file=FILE_OUT)
def build_config(config_template, config_file, data_dir, files_step2, par_file, kag, tsu, event_id):
    pass

### TASKS FOR STEP2 ###
## execution of simulator thysea using pycompss. (test)
@constraint(processors=[{'processorType':'CPU', 'computingUnits':'1'},
                        {'processorType':'GPU', 'computingUnits':'1'}])
@mpi(binary=tsunamiHySEA_bin, args="{{file_in}}", runner="mpirun", processes=gpus_per_exec, processes_per_node=gpus_per_node,working_dir="{{wdir}}")
@task(file_in=FILE_IN,returns=1)
def mpi_func(file_in, wdir):
     pass

@binary(binary=simulBS_bin, working_dir="{{wdir}}")
@task(sim_files_step2=FILE_OUT)
def build_structure(seistype, grid, hours, group, sim_files_step2, load_balancing, pois_ts_file, sim_template_file, sim_events_files, wdir):
    pass


#@task(log_file_sim=FILE_OUT)
@task(returns=1)
def extract_ts(out_ts,depth_file,ptf_file,simwdir,centinel):
     step2_extract_ts(out_ts,depth_file,ptf_file,simwdir)


#@task(log_file=FILE_OUT)
@task(returns=1)
def create_ptf_input(ptf_files,out_path,depth_file,log_file):
     step2_create_ptf_input(ptf_files,out_path,depth_file,log_file)


@task(ptf_files=COMMUTATIVE, config_file=FILE_IN)
def append_and_evaluate(ptf_files, ptf_file, args, config_file, sim_files_step1, out_step2_path, out_update_path, out_final, depth_file, log_file, sim_pois_ts, num_sims, kag, tsu, result_ext):
    args.cfg = config_file
    ptf_files.append(ptf_file)
    if (num_sims != 0) and (len(ptf_files) % num_sims == 0):
        step2_create_ptf_input(ptf_files, out_step2_path, depth_file, log_file)
        if kag>0:
            run_step_kagan(args,sim_files_step1,out_update_path)
            sim_files_input=out_update_path
        elif tsu>0:
            run_step_mare(args, sim_files_step1, out_update_path, sim_pois_ts, ptf_files)
            sim_files_input=out_update_path
        else:
            sim_files_input=sim_files_step1
        run_step3_init(args, sim_files_input, out_final, sim_pois_ts, ptf_files)


def build_steps_dirs(exec_dir, seistype):
     files_step1 = exec_dir + "/ptf_local_step1/"
     os.makedirs(files_step1)
     files_step2 = exec_dir + "/ptf_local_step2/"
     os.makedirs(files_step2)
     intermediate_files =  exec_dir + "/ptf_local_update/"
     os.makedirs(intermediate_files)
     files_step3 = exec_dir + "/ptf_local_step3/"
     os.makedirs(files_step3)
     step2_folder = exec_dir+"/Step2_"+seistype
     os.makedirs(step2_folder)
     return files_step1, files_step2, files_step3, intermediate_files

####################### MAIN SCRIPT ############################

if __name__ == '__main__':
    
     # reading arguments 
     
     args = parse_ptf_stdin() 
     exec_dir = args.run_path
     data_dir = args.data_path
     template_dir = args.templates_path
     par_file = args.parameters_file      # why are there 2 args for the par_file in run_bsc_mod ?
     if (exec_dir is None) or (data_dir is None) or (template_dir is None) or (par_file is None):
         print("One of these parameters is missing: run_path, data_path, templates_path, par_file" )
     else:  
         hours = args.hours                # the arg hours is missing in the run_bsc_mod.sh
         group_sims = int(args.group_sims)      # same
         print("Kagan: " + args.kagan_weight)
         print("Mare: " + args.mare_weight) 
         kag = int(args.kagan_weight)
         tsu = int(args.mare_weight)
         #cfg_file = args.cfg
         seistype = args.seistype
         event_id = args.event
         args.event = data_dir + "/earlyEst/" + event_id + "_stat.json"

         files_step1, files_step2, files_step3, intermediate_files = build_steps_dirs(exec_dir, seistype)
         
         ### Building Configuration
         config_file = exec_dir + "/ptf_main.config"
         config_template = template_dir + "/Step1_config_template_mod.txt"
         build_config(config_template, config_file, data_dir, files_step2, par_file, str(kag), str(tsu), event_id) 
         
         ### creation of the scenario ensemble ###
         sim_step1_events_file = step1_func(args, config_file, seistype, files_step1)
         
         depth_file = data_dir + "/regional_domain/bathy_grids/regional_domain_POIs_depth.dat"
         grid_file = data_dir + "/regional_domain/bathy_grids/regional_domain.grd" 
         pois_file = data_dir +"/regional_domain/POIs.txt"
         sim_template_file = template_dir + "/Step2_parfile_tmp.txt"
         log_file = files_step2 + "/Step2_"+seistype+"_failed.txt"
         sim_pois_ts = exec_dir + "/Step2_ts.dat"
         sim_files_step2 = exec_dir +"/sim_files.txt"
         
         ### Preparation of the files for HySEA simulation ###
         build_structure(seistype, grid_file, hours, gpus_per_node, sim_files_step2, load_balancing_bin, pois_file, sim_template_file, sim_step1_events_file, exec_dir)
         compss_wait_on_file(sim_files_step2) 
    
         ptf_files = []
         sims = 0
         ### HySEA simulations ###
         with open(sim_files_step2) as f:
             for line in f:
                # load balancing
                line=exec_dir+'/'+line.strip()
                print("Submitting execution for " +  line )
                result=mpi_func(line,exec_dir)
                
                with open(line) as fsim:
                     print('Entering fsim :', line)
                     for simline in fsim:
                         sims = sims + 1
                         simline = simline.strip()
                         print('Entering simline :', simline)
                         simwdir = os.path.dirname(simline)
                         print('sim dir',simwdir)
                         out_ts = exec_dir +"/"+simwdir+"/out_ts.nc"
                         ptf_file = exec_dir +"/"+simwdir+"/out_ts_ptf.nc"
                         ts_result = extract_ts(out_ts,depth_file,ptf_file,simwdir,result)
                         append_and_evaluate(ptf_files, ptf_file, args, config_file, files_step1, files_step2, intermediate_files, files_step3, depth_file, log_file, sim_pois_ts, group_sims, kag, tsu, ts_result)
                fsim.close()
         f.close()
         # Compute a final evaluation if final group was not multiple of group_sims
         if (group_sims == 0) or (sims % group_sims != 0):
              append_and_evaluate(ptf_files, ptf_file, args, config_file, files_step1, files_step2, intermediate_files, files_step3, depth_file, log_file, sim_pois_ts, 1, kag, tsu, ts_result)
    

