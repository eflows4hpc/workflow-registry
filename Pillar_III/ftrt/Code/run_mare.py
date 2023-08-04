#!/usr/bin/env python
#!/home/louise/miniconda3/bin/python3.8

# Import system modules
import os
import configparser
import hickle as hkl
import numpy       as np
from datetime import datetime

# Import functions from pyPTF modules
from ptf_preload             import load_PSBarInfo
from ptf_preload             import ptf_preload
from ptf_preload             import load_Scenarios_Reg
from ptf_load_event          import load_event_parameters
from ptf_load_event          import print_event_parameters
from ptf_parser              import update_cfg
from ptf_save                  import save_ptf_out
from ptf_ensemble_mare             import compute_ensemble_mare
from ptf_preload_curves        import load_hazard_values
from ptf_save                  import load_ptf_out

def step_mare(**kwargs):

    LongTermInfo     = kwargs.get('LongTermInfo', None)
    POIs             = kwargs.get('POIs', None)
    args             = kwargs.get('args', None)
    Config           = kwargs.get('cfg', None)
    event_parameters = kwargs.get('event_data', None)
    h_curve_files    = kwargs.get('h_curve_files', None)
    ptf_out          = kwargs.get('ptf_out', None)
    list_tmp_scen    = kwargs.get('list_tmp_scen', None)

    OR_EM=int(Config.get('Sampling','OR_EM'))
    MC_samp_scen=int(Config.get('Sampling','MC_samp_scen'))
    MC_samp_run=int(Config.get('Sampling','MC_samp_run'))
    RS_samp_scen=int(Config.get('Sampling','RS_samp_scen'))
    RS_samp_run=int(Config.get('Sampling','RS_samp_run'))
    EventID=Config.get('EventID','eventID')
    Mare_weights=int(Config.get('Sampling','Mare_weights'))
    TSU_path=Config.get('EventID','TSU_path')

    ### Loading of the tsunami data ###
    h5file = TSU_path
    TSU_file=hkl.load(h5file)
    tsu_data=TSU_file[EventID]
    Ntsu=len(tsu_data)

################### Monte Carlo sampling ########################
    
    if MC_samp_scen>0: 

       type_ens='MC'
       ### Calculation of the new ens ptf values without sampling
       samp_test='no_samp'
       samp_weight='weight'
       #print(tsu_data)
       half_tsu = int(len(tsu_data)/2)
       tsu_data_wei = tsu_data[0:half_tsu]
       #half_tsu_rand = random.sample(range(0,len(tsu_data)), half_tsu)
       #tsu_data_wei = tsu_data[half_tsu_rand]
       #print(tsu_data_wei)
       #print(half_tsu_rand)
       ptf_out = compute_ensemble_mare(cfg                = Config,
                                    event_parameters   = event_parameters,
                                    args               = args,
                                    ptf_out            = ptf_out,
                                    LongTermInfo       = LongTermInfo,
                                    pois               = POIs,
                                    h_curve_files      = h_curve_files,
                                    type_ens           = type_ens,
                                    samp_test          = samp_test,
                                    samp_weight          = samp_weight,
                                    list_tmp_scen      = list_tmp_scen,
                                    tsu_data           = tsu_data_wei)

################### Real sampling ########################

    if RS_samp_scen>0:

       type_ens='RS'
       ### Calculation of the new ens ptf values without sampling
       samp_test='no_samp'
       samp_weight='weight'
       half_tsu = int(len(tsu_data)/2)
       tsu_data_wei = tsu_data[0:half_tsu]
       ptf_out = compute_ensemble_mare(cfg                = Config,
                                    event_parameters   = event_parameters,
                                    args               = args,
                                    ptf_out            = ptf_out,
                                    LongTermInfo       = LongTermInfo,
                                    pois               = POIs,
                                    h_curve_files      = h_curve_files,
                                    type_ens           = type_ens,
                                    samp_test          = samp_test,
                                    samp_weight          = samp_weight,
                                    list_tmp_scen      = list_tmp_scen,
                                    tsu_data           = tsu_data_wei)


    print('End Tsunami reweight')

    return ptf_out



################################################################################################
#                                                                                              #
#                                  BEGIN                                                       #
################################################################################################

#def run_step1_init(config,event):
def run_step_mare(args,sim_files_step1,sim_files_update,sim_pois,ptf_files):

    # Read Stdin
    cfg_file        = args.cfg
    Config          = configparser.RawConfigParser()
    Config.read(cfg_file)
    Config          = update_cfg(cfg=Config, args=args)
    min_mag_message = float(Config.get('matrix','min_mag_for_message'))
    step2_mod = "SIM"
    pwd = os.getcwd()
    ############################################################
    #LOAD INFO FROM SPTHA
    PSBarInfo                                         = load_PSBarInfo(cfg=Config, args=args)
    # hazard_curves_files                               = load_hazard_values(cfg=Config, args=args, in_memory=True)
    Scenarios_PS, Scenarios_BS                        = load_Scenarios_Reg(cfg=Config, args=args, in_memory=True)
    LongTermInfo, POIs, Mesh, Region_files            = ptf_preload(cfg=Config, args=args)
    begin_of_time = datetime.utcnow()
    
    # gaga='/data/pyPTF/hazard_curves/glVal_BS_Reg032-E02352N3953E02776N3680.hdf5'
    
    # with h5py.File(gaga, "r") as f:
    #     a_group_key = list(f.keys())[0]
    #     datagaga = np.array(f.get(a_group_key))
    # print(np.shape(datagaga), datagaga.nbytes)
    
    #### Load event parameters then workflow and ttt are parallel
    #print('############################')
    # Load the event parameters from json file consumed from rabbit
    #event        = kwargs.get('event', None)
    event_parameters = load_event_parameters(event       = args.event,
                                             format      = args.event_format,
                                             routing_key = 'INT.QUAKE.CAT',
                                             args        = args,
                                             json_rabbit = None,
                                             cfg         = Config)
    print_event_parameters(dict=event_parameters, args = args)
   

    list_tmp_scen=np.zeros(len(ptf_files))
    for j in range(len(ptf_files)):
        line_tmp = ptf_files[j]
        print("line_tmp: "+ line_tmp)
        num_str = os.path.basename(os.path.dirname(line_tmp)).split("_")[2] 
        print("Num_str "+ str(num_str), flush=True)
        list_tmp_scen[j] = int(num_str)-1
        list_tmp_scen = list_tmp_scen.astype(int)
 
    ######################################################
    # Ensemble evaluation
    ptf_out = load_ptf_out(cfg=Config, args=args, event_parameters=event_parameters, sim_files=sim_files_step1)
    print("Sim POIs:" + str(sim_pois))
    h_curve_files    = load_hazard_values(cfg=Config, args=args, in_memory=True, ptf_out=ptf_out,step2_mod=step2_mod,POIs=POIs,sim_pois=sim_pois)

    ptf_out = step_mare(cfg                   = Config,
                           args                     = args,
                           POIs                     = POIs,
                           event_parameters         = event_parameters,
                           LongTermInfo             = LongTermInfo,
                           h_curve_files            = h_curve_files,
                           list_tmp_scen            = list_tmp_scen,
                           ptf_out                  = ptf_out)

    ######################################################
    # Save outputs
    print("Save pyPTF output")
    saved_files = save_ptf_out(cfg                = Config,
                               args               = args,
                               event_parameters   = event_parameters,
                               ptf                = ptf_out,
                               sim_files          = sim_files_update)
    
    #saved_files = save_ptf_dictionaries(cfg                = Config,
    #                                        args               = args,
    #                                        event_parameters   = event_parameters,
    #                                        ptf                = ptf_out,
    #                                        #status             = status,
    #                                        )
    #
    #
    #######################################################
    ## Make figures from dictionaries
    #print("Make pyPTF figures")
    #saved_files = make_ptf_figures(cfg                = Config,
    #                                   args               = args,
    #                                   event_parameters   = event_parameters,
    #                                   ptf                = ptf_out,
    #                                   saved_files        = saved_files)
    #
    #print("Save some extra usefull txt values")
    #saved_files = save_ptf_as_txt(cfg                = Config,
    #                                  args               = args,
    #                                  event_parameters   = event_parameters,
    #                                  ptf                = ptf_out,
    #                                  #status             = status,
    #                                  pois               = ptf_out['POIs'],
    #                                  #alert_levels       = ptf_out['alert_levels'],
    #                                  saved_files        = saved_files,
    #                                  #fcp                = fcp_merged,
    #                                  ensembleYN         = True
    #                                  )
    #
    #
    #
