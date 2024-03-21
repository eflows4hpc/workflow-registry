import os
import sys
import numpy as np
import configparser
import re
from datetime import datetime
import matplotlib.pyplot as plt

from ptf_preload               import ptf_preload
from ptf_preload               import load_Scenarios_Reg
from ptf_preload             import load_PSBarInfo
from ptf_load_event          import load_event_parameters
from ptf_load_event          import print_event_parameters
from ptf_parser                import update_cfg
from ptf_hazard_curves     import compute_hazard_curves
from ptf_hazard_curves_sim     import compute_hazard_curves_sim
from ptf_preload_curves        import load_hazard_values
from ptf_save                  import save_ptf_dictionaries
from ptf_save                  import load_ptf_out
from ptf_save                  import save_ptf_out_int
from ptf_figures               import make_figures 

def step3_hazard(**kwargs):

    Config                  = kwargs.get('Config', None)
    args                    = kwargs.get('args', None)
    POIs                    = kwargs.get('POIs', None)
    event_parameters        = kwargs.get('event_parameters', None)
    Scenarios_BS            = kwargs.get('Scenarios_BS', None)
    Scenarios_PS            = kwargs.get('Scenarios_PS', None)
    LongTermInfo            = kwargs.get('LongTermInfo', None)
    h_curve_files           = kwargs.get('h_curve_files', None)
    ptf_out                 = kwargs.get('ptf_out', None)
    list_tmp_scen           = kwargs.get('list_tmp_scen', None)

    OR_HC=int(Config.get('Sampling','OR_HC'))
    MC_samp_scen=int(Config.get('Sampling','MC_samp_scen'))
    MC_samp_run=int(Config.get('Sampling','MC_samp_run'))
    RS_samp_scen=int(Config.get('Sampling','RS_samp_scen'))
    RS_samp_run=int(Config.get('Sampling','RS_samp_run'))
    short_term_probability = ptf_out['short_term_probability']
    probability_scenarios = ptf_out['probability_scenarios']


    ######################################################
    # Compute hazard curves
    
    if OR_HC>0:
       print('Compute hazard curves at POIs')
       begin_of_utcc       = datetime.utcnow()
       hazard_curves = compute_hazard_curves(cfg                = Config,
                                             args               = args,
                                             pois               = POIs,
                                             event_parameters   = event_parameters,
                                             probability_scenarios  = probability_scenarios,
                                             Scenarios_BS       = Scenarios_BS,
                                             Scenarios_PS       = Scenarios_PS,
                                             LongTermInfo       = LongTermInfo,
                                             h_curve_files      = h_curve_files,
                                             short_term_probability = short_term_probability)
       end_of_utcc       = datetime.utcnow()
       diff_time        = end_of_utcc - begin_of_utcc
       print(" --> Building Hazard curves in Time [sec]: %s:%s" % (diff_time.seconds, diff_time.microseconds/1000))
       ######################################################      
       # in order to plot here add nested dict to ptf_out
       ptf_out['hazard_curves_original']          = hazard_curves

    ######################################################
    # Compute hazard curves

    if MC_samp_scen>0: 
       rangenid=len(ptf_out['new_ensemble_MC'])
       for Nid in range(rangenid):
           probability_scenarios = ptf_out['new_ensemble_MC'][Nid]
           print('Compute hazard curves at POIs')
           begin_of_utcc       = datetime.utcnow()
           hazard_curves = compute_hazard_curves_sim(cfg                = Config,
                                                 args               = args,
                                                 pois               = POIs,
                                                 event_parameters   = event_parameters,
                                                 probability_scenarios  = probability_scenarios,
                                                 Scenarios_BS       = Scenarios_BS,
                                                 Scenarios_PS       = Scenarios_PS,
                                                 LongTermInfo       = LongTermInfo,
                                                 list_tmp_scen      = list_tmp_scen,
                                                 h_curve_files      = h_curve_files,
                                                 short_term_probability = short_term_probability)
           end_of_utcc       = datetime.utcnow()
           diff_time        = end_of_utcc - begin_of_utcc
           print(" --> Building Hazard curves in Time [sec]: %s:%s" % (diff_time.seconds, diff_time.microseconds/1000))
           ######################################################

           # in order to plot here add nested dict to ptf_out

           ptf_out['hazard_curves_MC_%d'%Nid]          = hazard_curves

    ######################################################
    # Compute hazard curves

    if RS_samp_scen>0:
       rangenid=len(ptf_out['new_ensemble_RS'])
       for Nid in range(rangenid):
           if len(list_tmp_scen)<len(ptf_out['new_ensemble_RS'][Nid]['RealProbScenBS']):
              ProbScenBS_temp=ptf_out['new_ensemble_RS'][Nid]['RealProbScenBS'][list_tmp_scen]
              ptf_out['new_ensemble_RS'][Nid]['RealProbScenBS']=ProbScenBS_temp/np.sum(ProbScenBS_temp)
              probability_scenarios = ptf_out['new_ensemble_RS'][Nid]
           else:
              probability_scenarios = ptf_out['new_ensemble_RS'][Nid]

           print('Compute hazard curves at POIs')
           begin_of_utcc       = datetime.utcnow()
           hazard_curves = compute_hazard_curves_sim(cfg                = Config,
                                                 args               = args,
                                                 pois               = POIs,
                                                 event_parameters   = event_parameters,
                                                 probability_scenarios  = probability_scenarios,
                                                 Scenarios_BS       = Scenarios_BS,
                                                 Scenarios_PS       = Scenarios_PS,
                                                 LongTermInfo       = LongTermInfo,
                                                 list_tmp_scen      = list_tmp_scen,
                                                 h_curve_files      = h_curve_files,
                                                 short_term_probability = short_term_probability)
           end_of_utcc       = datetime.utcnow()
           diff_time        = end_of_utcc - begin_of_utcc
           print(" --> Building Hazard curves in Time [sec]: %s:%s" % (diff_time.seconds, diff_time.microseconds/1000))
           ######################################################

           # in order to plot here add nested dict to ptf_out

           ptf_out['hazard_curves_RS_%d'%Nid]          = hazard_curves

    ptf_out['list_ind_scen'] = list_tmp_scen

    return ptf_out

#                                  BEGIN                                                       #
################################################################################################

def run_step3_init(args,sim_files_step1,sim_files_step3,sim_pois,ptf_files,fail,conv_file,fig_path):

    ############################################################
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
    Scenarios_PS, Scenarios_BS                        = load_Scenarios_Reg(cfg=Config, args=args, in_memory=True)
    LongTermInfo, POIs, Mesh, Region_files            = ptf_preload(cfg=Config, args=args)
    
    #### Load event parameters then workflow and ttt are parallel
    #print('############################')
    print('Load event parameters')
    # Load the event parameters from json file consumed from rabbit
    event_parameters = load_event_parameters(event       = args.event,
                                             format      = args.event_format,
                                             routing_key = 'INT.QUAKE.CAT',
                                             args        = args,
                                             json_rabbit = None,
                                             cfg         = Config)
    
    print_event_parameters(dict=event_parameters, args = args)


    ### Identification of the failed simulations ###
    ### Creation of two lists of scenarios : 
    # 1) a no_fail one for calulating the HC without introducing the errors from the failed one
    no_fail = np.where(fail == 0)[0]
    ptf_no_fail = [ptf_files[i] for i in no_fail]
    list_tmp_scen=np.zeros(len(no_fail))
    for j in range(len(no_fail)):
        line_tmp = ptf_no_fail[j]
        line = re.findall('(\d+|[A-Za-z]+)', str(line_tmp[-24:-14]))
        list_tmp_scen[j] = int(line[1])-1
        list_tmp_scen = list_tmp_scen.astype(int)

    # 2) a yes_fail one for saving the scenarios id and writing a warning file
    yes_fail = np.where(fail == 3)[0]
    if len(yes_fail)>0:
       if len(yes_fail)==1:
          ptf_fail = ptf_files[int(yes_fail)]
          line_tmp = ptf_fail
          line = re.findall('(\d+|[A-Za-z]+)', str(line_tmp[-24:-14]))
          list_tmp_fail = int(line[1])-1
       elif len(yes_fail)>1:
          ptf_fail = [ptf_files[i] for i in yes_fail]
          list_tmp_fail=np.zeros(len(yes_fail))
          for j in range(len(yes_fail)):
              line_tmp = ptf_fail[j]
              line = re.findall('(\d+|[A-Za-z]+)', str(line_tmp[-24:-14]))
              list_tmp_fail[j] = int(line[1])-1
              list_tmp_fail = list_tmp_fail.astype(int)
       myfile = open(sim_files_step3+'/'+str(len(fail))+"_"+str(len(yes_fail))+"_failed_scenarios",'w')
       for Nscen in list_tmp_fail:
           myfile.write("%d\n"%(Nscen))
       myfile.close()
    else:
       myfile = open(sim_files_step3+'/'+str(len(fail))+"_no_failed_scenario",'w')
       myfile.write("No fails")
       myfile.close()


    ### Load STEP1 and STEP2 files ###
    ptf_out          = load_ptf_out(cfg=Config, args=args, event_parameters=event_parameters, sim_files=sim_files_step1)  #status= status,)
    h_curve_files    = load_hazard_values(cfg=Config, args=args, in_memory=True, ptf_out=ptf_out, step2_mod=step2_mod,POIs=POIs, sim_pois=sim_pois, list_tmp_scen=no_fail)
 
    ptf_out = step3_hazard(Config                   = Config,
                           args                     = args,
                           POIs                     = POIs,
                           event_parameters         = event_parameters,
                           Scenarios_BS             = Scenarios_BS,
                           Scenarios_PS             = Scenarios_PS,
                           LongTermInfo             = LongTermInfo,
                           h_curve_files            = h_curve_files,
                           list_tmp_scen            = list_tmp_scen,
    		           ptf_out                  = ptf_out)


    ### Updating the convergence file with the HC metrics of the new set of scenarios ###    
    Nid=0
    hazard_curves=ptf_out['hazard_curves_RS_%d'%Nid]
    mean_hc = np.mean(hazard_curves['hazard_curves_at_pois_mean'])
    std_hc = np.std(hazard_curves['hazard_curves_at_pois_mean'])
    myfile = open(conv_file,'a')
    myfile.write("%d %f %f\n"%(len(fail),mean_hc,std_hc))
    myfile.close()
   
    simplt=np.array([])
    meanplt=np.array([])
    stdplt=np.array([])
    with open(conv_file) as f:
         #lines = f.readlines()
         for (i,line) in enumerate(f):
             iline=np.array(line.split())
             simplt = np.append(simplt,int(iline[0]))
             meanplt = np.append(meanplt,float(iline[1]))
             stdplt = np.append(stdplt,float(iline[2]))

    fig, axs = plt.subplots(1,2)
    fig.suptitle('RS convergence')
    leg=['mean','stdv']
    fig.subplots_adjust(hspace=0.3)
    axs[0].plot(simplt,meanplt,color = 'C0',label='%s'%leg[0])
    axs[0].legend()
    axs[0].set_title("Convergence of the mean HC value", fontsize=10)
    axs[1].plot(simplt,stdplt,color = 'C1',label='%s'%leg[1])
    axs[1].legend()
    axs[1].set_title("Convergence of the std HC value", fontsize=10)
    plt.savefig(sim_files_step3+'Conv_%d_scenarios.png'%(len(fail)),bbox_inches='tight',dpi=300)

        
    len_tmp_scen = len(ptf_no_fail) 
    ######################################################
    # Save outputs
    print("Save pyPTF output")
    saved_files = save_ptf_out_int(cfg                = Config,
                                   args               = args,
                                   event_parameters   = event_parameters,
                                   ptf                = ptf_out,
                                   list_tmp_scen      = len_tmp_scen,
                                   sim_files          = sim_files_step3)

    file_hc = save_ptf_dictionaries(cfg                = Config,
                                    args               = args,
                                    event_parameters   = event_parameters,
                                    ptf                = ptf_out,
                                    list_tmp_scen      = len_tmp_scen,
                                    sim_files          = sim_files_step3)
    
    ## Make figures from dictionaries
    print("Make pyPTF figures")
    saved_files = make_figures(cfg                = Config,
                                       args               = args,
                                       event_parameters   = event_parameters,
                                       ptf                = ptf_out,
                                       file_hc            = file_hc,
                                       fig_path           = fig_path,
                                       fail               = fail)

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
    
