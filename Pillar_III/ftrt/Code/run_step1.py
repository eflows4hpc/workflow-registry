#!/usr/bin/env python
#!/home/louise/miniconda3/bin/python3.8

# Import system modules
import os
import configparser
import hickle as hkl
import numpy as np
from datetime import datetime

# Import functions from pyPTF modules
from ptf_preload             import load_PSBarInfo
from ptf_preload             import ptf_preload
from ptf_preload             import load_Scenarios_Reg
from ptf_load_event          import load_event_parameters
from ptf_load_event          import print_event_parameters
from ptf_parser              import update_cfg
from ptf_save                  import save_ptf_dictionaries
from ptf_save                  import save_ptf_out
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

def step1_ensembleEval(**kwargs):

    Scenarios_PS     = kwargs.get('Scenarios_PS', None)
    Scenarios_BS     = kwargs.get('Scenarios_BS', None)
    LongTermInfo     = kwargs.get('LongTermInfo', None)
    POIs             = kwargs.get('POIs', None)
    PSBarInfo        = kwargs.get('PSBarInfo', None)
    Mesh             = kwargs.get('Mesh', None)
    Region_files     = kwargs.get('Region_files', None)
    args             = kwargs.get('args', None)
    Config           = kwargs.get('cfg', None)
    event_parameters = kwargs.get('event_data', None)
    sim_files = kwargs.get('sim_files', None)

    OR_EM=int(Config.get('Sampling','OR_EM'))
    MC_samp_scen=int(Config.get('Sampling','MC_samp_scen'))
    MC_samp_run=int(Config.get('Sampling','MC_samp_run'))
    RS_samp_scen=int(Config.get('Sampling','RS_samp_scen'))
    RS_samp_run=int(Config.get('Sampling','RS_samp_run'))
    Kagan_weights=int(Config.get('Sampling','Kagan_weights'))
    NbrFM=int(Config.get('Sampling','NbrFM'))
    FM_path=Config.get('EventID','FM_path')
    EventID=Config.get('EventID','eventID')

    ptf_out = dict()


    ### Loading and selection of the focal mechanism ###
    h5file = FM_path
    print("h5file: " + FM_path)
    print("Event_ID: " + EventID)
    FM_file=hkl.load(h5file)
    print("FM_File: " + str(FM_file))
    totfm = len(FM_file[EventID])
    if NbrFM==0 or NbrFM>=totfm:
       focal_mech=FM_file[EventID]
    else:
       focal_mech=FM_file[EventID][0:NbrFM,:]
       focal_mech=np.array([focal_mech])
    Nfm=len(focal_mech)

    print('############## Initial ensemble #################')

    print('Build ellipsoids objects')
    ellipses = build_ellipsoid_objects(event = event_parameters,
                                       cfg   = Config,
                                       args  = args)


    print('Conversion to utm')
    LongTermInfo, POIs, PSBarInfo = conversion_to_utm(longTerm  = LongTermInfo,
                                                      Poi       = POIs,
                                                      event     = event_parameters,
                                                      PSBarInfo = PSBarInfo)

    ##########################################################
    # Set separation of lambda BS-PS
    print('Separation of lambda BS-PS')
    lambda_bsps = load_lambda_BSPS(cfg                   = Config,
                                   args                  = args,
                                   event_parameters      = event_parameters,
                                   LongTermInfo          = LongTermInfo)


    lambda_bsps = separation_lambda_BSPS(cfg              = Config,
                                         args             = args,
                                         event_parameters = event_parameters,
                                         lambda_bsps      = lambda_bsps,
                                         LongTermInfo     = LongTermInfo,
                                         mesh             = Mesh)

    #print(lambda_bsps['regionsPerPS'])
    #sys.exit()
    ##########################################################
    # Pre-selection of the scenarios
    #
    # Magnitude: First PS then BS
    # At this moment the best solution is to insert everything into a dictionary (in matlab is the PreSelection structure)
    print('Pre-selection of the Scenarios')
    pre_selection = pre_selection_of_scenarios(cfg                = Config,
                                               args               = args,
                                               event_parameters   = event_parameters,
                                               LongTermInfo       = LongTermInfo,
                                               PSBarInfo          = PSBarInfo,
                                               ellipses           = ellipses)

    if(pre_selection == False):
        list_tmp_scen = 0
        ptf_out = save_ptf_dictionaries(cfg                = Config,
                                        args               = args,
                                        event_parameters   = event_parameters,
                                        sim_files          = sim_files,
                                        list_tmp_scen      = list_tmp_scen,
                                        status             = 'end')
        return False


    if OR_EM>0:

        print('############## Initial ensemble #################')

        ##########################################################
        # COMPUTE PROB DISTR
        #
        #    Equivalent of shortterm.py with output: node_st_probabilities
        #    Output: EarlyEst.MagProb, EarlyEst.PosProb, EarlyEst.DepProb, EarlyEst.DepProb, EarlyEst.BarProb, EarlyEst.RatioBSonTot
        print('Compute short term probability distribution')

        short_term_probability  = short_term_probability_distribution(cfg                = Config,
                                                                      args               = args,
                                                                      event_parameters   = event_parameters,
                                                                      LongTermInfo       = LongTermInfo,
                                                                      PSBarInfo          = PSBarInfo,
                                                                      lambda_bsps        = lambda_bsps,
                                                                      pre_selection      = pre_selection)

        if(short_term_probability == True):
            list_tmp_scen = 0
            ptf_out = save_ptf_dictionaries(cfg                = Config,
                                            args               = args,
                                            event_parameters   = event_parameters,
                                            list_tmp_scen      = list_tmp_scen,
                                            status             = 'end')
            return False

        ##COMPUTE PROBABILITIES SCENARIOS: line 840
        print('Compute Probabilities scenarios')
        probability_scenarios = compute_probability_scenarios(cfg                = Config,
                                                              args               = args,
                                                              event_parameters   = event_parameters,
                                                              LongTermInfo       = LongTermInfo,
                                                              PSBarInfo          = PSBarInfo,
                                                              lambda_bsps        = lambda_bsps,
                                                              pre_selection      = pre_selection,
                                                              regions            = Region_files,
                                                              short_term         = short_term_probability,
                                                              Scenarios_PS       = Scenarios_PS)

    else:

        short_term_probability=0.0
        probability_scenarios=0.0


################### Monte Carlo sampling ########################
    
    if MC_samp_scen>0: 
       print('############## Monte Carlo sampling #################')
       sampled_ensemble_MC = compute_ensemble_sampling_MC(cfg                = Config,
                                                 args               = args,
                                                 event_parameters   = event_parameters,
                                                 LongTermInfo       = LongTermInfo,
                                                 PSBarInfo          = PSBarInfo,
                                                 lambda_bsps        = lambda_bsps,
                                                 pre_selection      = pre_selection,
                                                 regions            = Region_files,
                                                 short_term         = short_term_probability,
                                                 Scenarios_PS       = Scenarios_PS,
                                                 proba_scenarios    = probability_scenarios)
       ptf_out['new_ensemble_MC']           = sampled_ensemble_MC


       for Nid in range(MC_samp_run):
           MC_samp_scen=len(ptf_out['new_ensemble_MC'][Nid]['par_scenarios_bs'][:,0])
           par=np.zeros((11))
           myfile = open(sim_files+"Step1_scenario_list_BS.txt",'w')
           for Nscen in range(MC_samp_scen):
               par[:]=ptf_out['new_ensemble_MC'][Nid]['par_scenarios_bs'][Nscen,:]
               myfile.write("%d %f %f %f %f %f %f %f %f %f %f\n"%(Nscen,par[1],par[2],par[3],par[4],par[5],par[6],par[7],par[8],par[9],par[10]))
           myfile.close()


################### Real sampling ########################

    if RS_samp_scen>0:
       print('############## RS sampling #################')
       sampled_ensemble_RS = compute_ensemble_sampling_RS(cfg                = Config,
                                                 args               = args,
                                                 event_parameters   = event_parameters,
                                                 LongTermInfo       = LongTermInfo,
                                                 PSBarInfo          = PSBarInfo,
                                                 lambda_bsps        = lambda_bsps,
                                                 pre_selection      = pre_selection,
                                                 regions            = Region_files,
                                                 short_term         = short_term_probability,
                                                 Scenarios_PS       = Scenarios_PS,
                                                 proba_scenarios    = probability_scenarios)

       ptf_out['new_ensemble_RS']           = sampled_ensemble_RS

       for Nid in range(RS_samp_run):
           #RS_samp_scen=len(ptf_out['new_ensemble_RS'][0]['real_par_scenarios_bs'])
           par=np.zeros((11))
           myfile = open(sim_files+"Step1_scenario_list_BS.txt",'w')
           for Nscen in range(RS_samp_scen):
               par[:]=ptf_out['new_ensemble_RS'][0]['real_par_scenarios_bs'][Nscen]
               myfile.write("%d %f %f %f %f %f %f %f %f %f %f\n"%(Nscen,par[1],par[2],par[3],par[4],par[5],par[6],par[7],par[8],par[9],par[10]))
           myfile.close()

    #if(probability_scenarios == False):
    #    print( "--> No Probability scenarios found. Save and Exit")
    #    ptf_out = save_ptf_dictionaries(cfg                = Config,
    #                                    args               = args,
    #                                    event_parameters   = event_parameters,
    #                                    sim_files          = sim_files,
    #                                    status             = 'end')
    #    return False


    # in order to plot here add nested dict to ptf_out
    ptf_out['short_term_probability'] = short_term_probability
    ptf_out['event_parameters']       = event_parameters
    ptf_out['probability_scenarios']  = probability_scenarios
    ptf_out['POIs']                   = POIs

    print('End pyPTF')

    return ptf_out



################################################################################################
#                                                                                              #
#                                  BEGIN                                                       #
################################################################################################

#def run_step1_init(config,event):
def run_step1_init(args,sim_files):

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
    print(cfg_file)
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
    
    begin_of_time = datetime.utcnow()
    
    # gaga='/data/pyPTF/hazard_curves/glVal_BS_Reg032-E02352N3953E02776N3680.hdf5'
    
    # with h5py.File(gaga, "r") as f:
    #     a_group_key = list(f.keys())[0]
    #     datagaga = np.array(f.get(a_group_key))
    # print(np.shape(datagaga), datagaga.nbytes)
    end_of_time = datetime.utcnow()
    diff_time        = end_of_time - begin_of_time
    print("--> Execution Time [sec]: %s:%s" % (diff_time.seconds, diff_time.microseconds))
    #sys.exit()
    
    
    begin_of_time = datetime.utcnow()
    
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

    list_tmp_scen = int(Config.get('Sampling','RS_samp_scen'))
    ptf_out = save_ptf_dictionaries(cfg                = Config,
                                    args               = args,
                                    event_parameters   = event_parameters,
                                    sim_files          = sim_files,
                                    list_tmp_scen      = list_tmp_scen,
                                    status             = 'new')
    
    
    ######################################################
    # Ensemble evaluation
    
    
    ptf_out = step1_ensembleEval(Scenarios_PS = Scenarios_PS,
                                            Scenarios_BS = Scenarios_BS,
                                            LongTermInfo = LongTermInfo,
                                            POIs         = POIs,
                                            PSBarInfo    = PSBarInfo,
                                            Mesh         = Mesh,
                                            Region_files = Region_files,
                                            #h_curve_files= hazard_curves_files,
                                            args         = args,
                                            cfg          = Config,
                                            event_data   = event_parameters,
                                            sim_files    = sim_files)
    
    
    ######################################################
    # Save outputs
    print("Save pyPTF output")
    saved_files = save_ptf_out(cfg                = Config,
                               args               = args,
                               event_parameters   = event_parameters,
                               ptf                = ptf_out,
                               sim_files    = sim_files)
    
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
