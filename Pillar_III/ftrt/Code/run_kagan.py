#!/usr/bin/env python
#!/home/louise/miniconda3/bin/python3.8

# Import system modules
import os
import sys
import configparser
import hickle as hkl
import numpy       as np

# adding the path to find some modules
#sys.path.append('Step1_EnsembleDef_python/py')
pathmainfolder=sys.argv[0]
sys.path.append(pathmainfolder)
sys.path.append(pathmainfolder+'/../Common/py/')
sys.path.append('/gpfs/projects/bsc44/bsc44973/PTF_WF_test/Commons/py')

# Import functions from pyPTF modules
from ptf_preload             import load_PSBarInfo
from ptf_preload             import ptf_preload
from ptf_preload             import load_Scenarios_Reg
from ptf_load_event          import load_event_parameters
from ptf_load_event          import print_event_parameters
from ptf_parser              import parse_ptf_stdin
from ptf_parser              import update_cfg
from ptf_save                import save_ptf_dictionaries
from ptf_save                import save_ptf_dictionaries
from ptf_save                import save_ptf_as_txt
from ptf_save                  import save_ptf_dictionaries
from ptf_save                  import save_ptf_out
from ptf_ellipsoids          import build_location_ellipsoid_objects
from ptf_ellipsoids          import build_ellipsoid_objects
from ptf_mix_utilities        import conversion_to_utm
from ptf_lambda_bsps_load      import load_lambda_BSPS
from ptf_lambda_bsps_sep       import separation_lambda_BSPS
from ptf_pre_selection         import pre_selection_of_scenarios
from ptf_short_term                 import short_term_probability_distribution
from ptf_probability_scenarios      import compute_probability_scenarios
from ptf_ensemble_sampling_MC       import compute_ensemble_sampling_MC
from ptf_ensemble_sampling_RS       import compute_ensemble_sampling_RS
from ptf_ensemble_kagan             import compute_ensemble_kagan
from ptf_ensemble_mare             import compute_ensemble_mare
from ptf_hazard_curves         import compute_hazard_curves
from ptf_preload_curves        import load_hazard_values
from ptf_save                  import save_ptf_dictionaries
from ptf_save                  import load_ptf_out

def step_kagan(**kwargs):

    args             = kwargs.get('args', None)
    Config           = kwargs.get('cfg', None)
    event_parameters = kwargs.get('event_data', None)
    ptf_out          = kwargs.get('ptf_out', None)

    MC_samp_scen=int(Config.get('Sampling','MC_samp_scen'))
    RS_samp_scen=int(Config.get('Sampling','RS_samp_scen'))
    Kagan_weights=int(Config.get('Sampling','Kagan_weights'))
    NbrFM=int(Config.get('Sampling','NbrFM'))
    FM_path=Config.get('EventID','FM_path')
    EventID=Config.get('EventID','eventID')

    ### Loading and selection of the focal mechanism ###
    h5file = FM_path
    FM_file=hkl.load(h5file)
    totfm = len(FM_file[EventID])
    if NbrFM==0 or NbrFM>=totfm:
       focal_mech=FM_file[EventID]
    else:
       focal_mech=FM_file[EventID][0:NbrFM,:]
       focal_mech=np.array([focal_mech])
    Nfm=len(focal_mech)

    if MC_samp_scen>0: 
       print('############## Monte Carlo sampling #################')

       type_ens='MC'
       ptf_out = compute_ensemble_kagan(cfg                = Config,
                                        args               = args,
                                        event_parameters   = event_parameters,
                                        ptf_out            = ptf_out,
                                        focal_mechanism    = focal_mech,
                                        type_ens           = type_ens)


################### Real sampling ########################

    if RS_samp_scen>0:

       type_ens='RS'
       ptf_out = compute_ensemble_kagan(cfg                = Config,
                                        args               = args,
                                        event_parameters   = event_parameters,
                                        ptf_out            = ptf_out,
                                        focal_mechanism    = focal_mech,
                                        type_ens           = type_ens)


       
    print('End Kagan reweight')

    return ptf_out



################################################################################################
#                                                                                              #
#                                  BEGIN                                                       #
################################################################################################

#def run_step1_init(config,event):
def run_step_kagan(args,sim_files_step1,sim_files_update):

    ############################################################
    # Read Stdin
    #print('\n')
    #args=parse_ptf_stdin()
    #args           = kwargs.get('args', None)    

    ############################################################
    # Initialize and load configuaration file
    #cfg_file        = args.cfg
    #cfg_file        = kwargs.get('config', None)
    #event           = kwargs.get('event', None)
    cfg_file        = args.cfg

    Config          = configparser.RawConfigParser()
    Config.read(cfg_file)
    Config          = update_cfg(cfg=Config, args=args)
    min_mag_message = float(Config.get('matrix','min_mag_for_message'))

    pwd = os.getcwd()
    
    ############################################################
    #LOAD INFO FROM SPTHA
    PSBarInfo                                         = load_PSBarInfo(cfg=Config, args=args)
    # hazard_curves_files                               = load_hazard_values(cfg=Config, args=args, in_memory=True)
    Scenarios_PS, Scenarios_BS                        = load_Scenarios_Reg(cfg=Config, args=args, in_memory=True)
    LongTermInfo, POIs, Mesh, Region_files            = ptf_preload(cfg=Config, args=args)
    
    #### Load event parameters then workflow and ttt are parallel
    #print('############################')
    print('Load event parameters')
    # Load the event parameters from json file consumed from rabbit
    #event        = kwargs.get('event', None)
    event_parameters = load_event_parameters(event       = args.event,
                                             format      = args.event_format,
                                             routing_key = 'INT.QUAKE.CAT',
                                             args        = args,
                                             json_rabbit = None,
                                             cfg         = Config)
    print_event_parameters(dict=event_parameters, args = args)
    
    ######################################################
    # Ensemble evaluation
    
    ptf_out = load_ptf_out(cfg=Config, args=args, event_parameters=event_parameters, sim_files=sim_files_step1)
 
    ptf_out = step_kagan(ptf_out      = ptf_out,
                                 args         = args,
                                 cfg          = Config,
                                 event_data   = event_parameters)
    
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
    #                                        sim_files          = sim_files_update)
 
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
