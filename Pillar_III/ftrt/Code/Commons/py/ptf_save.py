import os
import h5py
import numpy as np
import pathlib
import hickle as hkl

def save_ptf_as_txt(**kwargs):

    Config                = kwargs.get('cfg', None)
    args                  = kwargs.get('args', None)
    ee                    = kwargs.get('event_parameters', None)
    ptf                   = kwargs.get('ptf', None)
    status                = kwargs.get('status', None)
    pois                  = kwargs.get('pois', None)
    alert_levels          = kwargs.get('alert_levels', None)
    saved_files           = kwargs.get('saved_files', None)
    fcp                   = kwargs.get('fcp', None)
    ensembleYN            = kwargs.get('ensembleYN', False)
    hazardCurvesYN        = kwargs.get('hazardCurvesYN', False)
    alertLevelsYN            = kwargs.get('alertLevelsYN', False)    

    sigma          = float(Config.get('Settings','nSigma'))
    out_f          = saved_files['hazard_poi_table']
    out_p          = saved_files['event_path']
    out_al_poi     = saved_files['al_poi_table']
    out_al_fcp     = saved_files['al_fcp_table']
    # p_levels       = alert_levels['probability']['probability_levels']

    out_ens_bs_par = saved_files['ensemble_bs_par']
    out_ens_bs_prob= saved_files['ensemble_bs_prob']
    out_ens_ps_par = saved_files['ensemble_ps_par']
    out_ens_ps_prob= saved_files['ensemble_ps_prob']
    out_new_ens_bs_par= saved_files['sampled_ensemble_bs_par']
    out_new_ens_bs_prob= saved_files['sampled_ensemble_bs_prob']

    ## Save ensemble parameters 
    if(ensembleYN): 
            #print(ptf['probability_scenarios']['ProbScenBS'])
            print(" --> Save ensemble - BS parameters in %s " % (out_p + os.sep +  out_ens_bs_par))
            with open(out_p + os.sep + out_ens_bs_par, 'w') as f:

                partable = ptf['probability_scenarios']['par_scenarios_bs']
                f.write("region, magnitude, lon, lat, depth of the top, strike, dip, rake, area (km2), length (km), average slip (m)\n")
                for i in range(len(partable)):
                    partmp = partable[i]
                    p = ' '.join(['%.2f' % (partmp[j]) for j in range(len(partmp))])
                    line = p + '\n'
                    f.write(line)

            f.close()

            print(" --> Save ensemble - BS probabilities in %s " % (out_p + os.sep +  out_ens_bs_prob))
            with open(out_p + os.sep + out_ens_bs_prob, 'w') as f:
                prob=ptf['probability_scenarios']['ProbScenBS']

                f.write("# BS probability\n")
                for i in range(len(prob)):
                    line = '%.5e \n' % (prob[i])
                    f.write(line)

            f.close()

            print(" --> Save ensemble - PS parameters in %s " % (out_p + os.sep +  out_ens_ps_par))
            with open(out_p + os.sep + out_ens_ps_par, 'w') as f:

                partable = ptf['probability_scenarios']['par_scenarios_ps']
                f.write("region, magnitude, lon, lat, slip model (Mourotani/Strasser, prop yes/no, rigidity), slip disribution (0: uniform; n > 0: number of samplings)\n")
                for i in range(len(partable)):
                    partmp = partable[i]
                    p = ' '.join(['%.2f' % (partmp[j]) for j in range(len(partmp))])
                    line = p + '\n'
                    f.write(line)

            f.close()

            print(" --> Save ensemble - PS probabilities in %s " % (out_p + os.sep +  out_ens_ps_prob))
            with open(out_p + os.sep + out_ens_ps_prob, 'w') as f:
                prob=ptf['probability_scenarios']['ProbScenPS']

                f.write("# PS probability")
                for i in range(len(prob)):
                    line = '%.5e \n' % (prob[i])
                    f.write(line)

            f.close()

            #Nsamp=np.arange(1,50,10)
            #Nsamp=[]#np.ones(100)*50 #Nsamp=[]#2,5,10,20,40,60,80,100,200,400,600,800,1000,5000,10000,15000,20000,50000]
            #Nid=0
            #for N in Nsamp:
            #    print(" --> Save ensemble - Sampled BS parameters in %s " % (out_p + os.sep + "Samp_" + str(N) + "_" + out_new_ens_bs_par))
            #    with open(out_p + os.sep + "Samp_" + str(N)+ "_" + out_new_ens_bs_par, 'w') as f:
            #        partable = ptf['new_ensemble'][Nid]['par_scenarios_bs']
            #        f.write("region, magnitude, lon, lat, depth of the top, strike, dip, rake, area (km2), length (km), average slip (m)\n")
            #        for i in range(len(partable)):
            #            partmp = partable[i]
            #            p = ' '.join(['%.2f' % (partmp[j]) for j in range(len(partmp))])
            #            line = p + '\n'
            #            f.write(line)
            #    Nid=Nid+1

            #f.close()

            #print(" --> Save ensemble - Sampled BS probabilities in %s " % (out_p + os.sep +  out_new_ens_bs_prob))
            #with open(out_p + os.sep + out_new_ens_bs_prob, 'w') as f:

            #    prob=ptf['new_ensemble']['prob_scenarios_bs']
            #    f.write("# Sampled BS probability\n")
            #    for i in range(len(prob)):
            #        line = '%.5e \n' % (prob[i])
            #        f.write(line)

            #f.close()

    ## Save table hazard values per poi
    if(hazardCurvesYN):
            print(" --> Save table hazard values at pois %s" % (out_p + os.sep +  out_f))
            with open(out_p + os.sep + out_f, 'w') as f:
                p = ' '.join(['%.2f' % (p_levels[n]) for n in range(len(p_levels))])
                h0 = "# pyptf hazard map values [m] at pois - Sigma: %.2f\n" % (sigma)
                h1 = "# lat lon best average probabiliy(1-p): %s\n" % (p)
                f.write(h0)
                f.write(h1)
                for i in range(len(pois['selected_pois'])):

                    lat  = pois['selected_lat'][i]
                    lon  = pois['selected_lon'][i]
                    best = alert_levels['best']['level_values'][i]
                    mean = alert_levels['average']['level_values'][i]
                    prob = alert_levels['probability']['level_values'][:,:][i]
                    ss = ' '.join(['%.6e' % (prob[n]) for n in  range(len(prob))])
                    aa = "%-7.3f %-7.3f %e %e %s\n" % (lat, lon, best, mean, ss)
                    f.write(aa)

            f.close()

    ## Save table alert levels per poi
    if(alertLevelsYN):
            print(" --> Save table alert levels at pois %s" % (out_p + os.sep +  out_al_poi))
            with open(out_p + os.sep + out_al_poi, 'w') as f:
                p = ' '.join(['%.2f' % (p_levels[n]) for n in range(len(p_levels))])
                h0 = "# pyptf alert levels at pois - Sigma: %.2f\n" % (sigma)
                h1 = "# lat lon matrix best average probabiliy(1-p): %s\n" % (p)
                f.write(h0)
                f.write(h1)
                for i in range(len(pois['selected_pois'])):
                    lat  = pois['selected_lat'][i]
                    lon  = pois['selected_lon'][i]
                    matrix = alert_levels['matrix_poi']['level_type'][i]
                    best = alert_levels['best']['level_type'][i]
                    mean = alert_levels['average']['level_type'][i]
                    prob = alert_levels['probability']['level_type'][i]
                    ss = ' '.join(['%1d' % (prob[n]) for n in  range(len(prob))])
                    aa = "%-7.3f %-7.3f %1d %1d %1d %s\n" % (lat, lon, matrix, best, mean, ss)
                    f.write(aa)

            f.close()

            print(" --> Save table alert levels at fcp %s" % (out_p + os.sep +  out_al_fcp))
            with open(out_p + os.sep + out_al_fcp, 'w') as f:
                p = ' '.join(['%.2f' % (p_levels[n]) for n in range(len(p_levels))])
                h0 = "# decision matrix and pyptf alert levels at fcp - Sigma: %.2f\n" % (sigma)
                h1 = "# lat lon DM matrix-ptf best average probabiliy(1-p): %s fcp_name fcp_state \n" % (p)
                f.write(h0)
                f.write(h1)
                for i in range(len(fcp['data'])):

                    name  = fcp['data'][i]['name']
                    state = fcp['data'][i]['state']
                    lat   = fcp['data'][i]['lat']
                    lon   = fcp['data'][i]['lon']
                    dm    = fcp['data'][i]['matrix_fcp_alert_type']
                    if (isinstance(fcp['data'][i]['ptf_fcp_alert_type'], int) == False):
                        matrix = fcp['data'][i]['ptf_fcp_alert_type'][-3]
                        best  = fcp['data'][i]['ptf_fcp_alert_type'][-2]
                        mean  = fcp['data'][i]['ptf_fcp_alert_type'][-1]
                        prob  = fcp['data'][i]['ptf_fcp_alert_type'][0:len(p_levels)]
                    else:
                        best = 0
                        mean = 0
                        prob = np.zeros(len(p_levels))
                    ss = ' '.join(['%1d' % (prob[n]) for n in  range(len(prob))])
                    aa = "%-7.3f %-7.3f %1s %1d %1d %1d %s %-25s %-20s \n" % (lat, lon, dm, matrix, best, mean, ss,name, state)
                    f.write(aa)

            f.close()

    return saved_files

def define_save_path(**kwargs):

    Config         = kwargs.get('cfg', None)
    args           = kwargs.get('args', None)
    event          = kwargs.get('event', None)
    save_d         = kwargs.get('dictionary', None)
    sim_files      = kwargs.get('sim_files',None)

    if(save_d == None):
        out = dict()
    else:
        out = save_d

    save_main_path = Config.get('save_ptf','save_main_path')
    save_sub_path  = Config.get('save_ptf','save_sub_path')

    if(args.save_main_path != None):
        save_main_path = args.save_main_path

    if(args.save_sub_path != None):
        save_sub_path =  args.save_sub_path

    save_path = save_main_path #+ os.sep + save_sub_path


    ro_main_path = save_path #+ os.sep + str(event['ot_year']) + os.sep + str(event['ot_month'])
    #ev_path_name = str(event['ot_year']) + str(event['ot_month']) + str(event['ot_day']) + '_' + \
    #               str(event['eventid']) + '_' + str(event['version']) + '_' + event['area']

    ev_path_name = str(event['ot_year']) + str(event['ot_month']) + str(event['ot_day']) + '_' + \
                   str(event['eventid']) + '_' + str(event['version']) + '_' + event['area']


#    out['event_path'] = ro_main_path + os.sep + ev_path_name
#    out['event_path'] = ro_main_path 
    out['event_path'] = sim_files

    return out

def define_file_names(**kwargs):

    Config         = kwargs.get('cfg', None)
    args           = kwargs.get('args', None)
    event          = kwargs.get('event', None)
    save_d         = kwargs.get('dictionary', None)

    MC_samp_scen=int(Config.get('Sampling','MC_samp_scen'))
    RS_samp_scen=int(Config.get('Sampling','RS_samp_scen'))

#    root_name = str(event['ot_year']) + str(event['ot_month']) + str(event['ot_day']) + '_' + \
#                str(event['eventid']) + '_' + str(event['version']) + '_' + event['area']
#    root_name = str(event['ot_year']) + str(event['ot_month']) + str(event['ot_day']) + '_' + event['area']
#    hazard_curves = root_name + '_' + Config.get('save_ptf','hazard_curves')
#    pois          = root_name + '_' + Config.get('save_ptf','pois')
#    event         = root_name + '_' + Config.get('save_ptf','event_parameters')
#    alert_levels  = root_name + '_' + Config.get('save_ptf','alert_levels')
#    poi_html_map  = root_name + '_' + Config.get('save_ptf','poi_html_map')
#    json_file     = root_name + '_' + Config.get('save_ptf','message_dict')
#    h_poi_table   = root_name + '_' + Config.get('save_ptf','table_hazard_poi')
#    al_poi_table  = root_name + '_' + Config.get('save_ptf','table_alert_level_poi')
#    al_fcp_table  = root_name + '_' + Config.get('save_ptf','table_alert_level_fcp')

    root_name = ''
    hazard_curves_original = Config.get('save_ptf','hazard_curves_original')
    pois          = Config.get('save_ptf','pois')
    event         = Config.get('save_ptf','event_parameters')
    alert_levels  = Config.get('save_ptf','alert_levels')
    poi_html_map  = Config.get('save_ptf','poi_html_map')
    json_file     = Config.get('save_ptf','message_dict')
    h_poi_table   = Config.get('save_ptf','table_hazard_poi')
    al_poi_table  = Config.get('save_ptf','table_alert_level_poi')
    al_fcp_table  = Config.get('save_ptf','table_alert_level_fcp')

    save_d['hazard_curves_original']    = hazard_curves_original
    if MC_samp_scen>0:
        hazard_curves_MC = Config.get('save_ptf','hazard_curves_MC')
        save_d['hazard_curves_MC']    = hazard_curves_MC
    if RS_samp_scen>0:
        hazard_curves_RS = Config.get('save_ptf','hazard_curves_RS')
        save_d['hazard_curves_RS']    = hazard_curves_RS
    
    save_d['pois']             = pois
    save_d['event_parameters'] = event
    save_d['alert_levels']     = alert_levels
    save_d['poi_html_map']     = poi_html_map
    save_d['json_file']        = json_file + '.json'
    save_d['hazard_poi_table'] = h_poi_table + '.txt'
    save_d['al_poi_table']     = al_poi_table + '.txt'
    save_d['al_fcp_table']     = al_fcp_table + '.txt'

   
#    ens_bs_par    = root_name + '_' + Config.get('save_ptf','table_ensemble_bs_par')
#    ens_bs_prob   = root_name + '_' + Config.get('save_ptf','table_ensemble_bs_prob')
#    ens_ps_par    = root_name + '_' + Config.get('save_ptf','table_ensemble_ps_par')
#    ens_ps_prob   = root_name + '_' + Config.get('save_ptf','table_ensemble_ps_prob')
    ens_bs_par    = Config.get('save_ptf','table_ensemble_bs_par')
    ens_bs_prob   = Config.get('save_ptf','table_ensemble_bs_prob')
    ens_ps_par    = Config.get('save_ptf','table_ensemble_ps_par')
    ens_ps_prob   = Config.get('save_ptf','table_ensemble_ps_prob')
    samp_ens_bs_par    = Config.get('save_ptf','table_sampled_ensemble_bs_par')
    samp_ens_bs_prob    = Config.get('save_ptf','table_sampled_ensemble_bs_prob')
    save_d['ensemble_bs_par']   = ens_bs_par + '.txt'
    save_d['ensemble_bs_prob']  = ens_bs_prob + '.txt' 
    save_d['ensemble_ps_par']   = ens_ps_par + '.txt' 
    save_d['ensemble_ps_prob']  = ens_ps_prob + '.txt'
    save_d['sampled_ensemble_bs_prob']  = samp_ens_bs_prob + '.txt'
    save_d['sampled_ensemble_bs_par']  = samp_ens_bs_par + '.txt'


    return save_d

def save_ptf_dictionaries(**kwargs):


    Config                = kwargs.get('cfg', None)
    args                  = kwargs.get('args', None)
    ee                    = kwargs.get('event_parameters', None)
    ptf                   = kwargs.get('ptf', None)
    status                = kwargs.get('status', None)
    sim_files             = kwargs.get('sim_files', None)
    list_tmp_scen         = kwargs.get('list_tmp_scen', None)

    OR_HC=int(Config.get('Sampling','OR_HC'))
    MC_samp_scen=int(Config.get('Sampling','MC_samp_scen'))
    RS_samp_scen=int(Config.get('Sampling','RS_samp_scen'))
    format_hc = Config.get('save_ptf','save_format_hc')
    inter = list_tmp_scen

    if args.save_format != None:
        format = args.save_format
    else:
        format    = Config.get('save_ptf','save_format')

    save_dict = define_save_path(cfg = Config,args = args, event = ee, sim_files = sim_files)

    save_dict = define_file_names(cfg = Config,args = args, event = ee, dictionary=save_dict)

    #######################
    if(status == 'new'):
       print(" --> Create %s" % save_dict['event_path'])
       pathlib.Path(save_dict['event_path']).mkdir(parents=True, exist_ok=True)

       return True

    #######################
    if(status == 'end'):
        print(" --> Save event parameters in file (%s-file format) %s.%s" % (format, save_dict['event_parameters'],format))
        if(format == 'npy'):
            np.save(save_dict['event_path'] + os.sep + save_dict['event_parameters'],ee,               allow_pickle=True)
        if(format == 'hdf5'):
            h5file = save_dict['event_path'] + os.sep + save_dict['event_parameters'] + '.hdf5'
            hkl.dump(ee, h5file, mode='w')

        return True


    #######################
    if(ptf != False):
        if OR_HC>0:
           hazard_curves_original         = ptf['hazard_curves_original']
        probability_scenarios = ptf['probability_scenarios']
        pois                  = ptf['POIs']
        if MC_samp_scen>0:
           new_ens_MC = {}
           rangemc=len(ptf['new_ensemble_MC'])
           hazard_curves_mc={}
           for Nid in range(rangemc):
               new_ens_MC['%d'%Nid] = ptf['new_ensemble_MC'][Nid]
               hazard_curves_mc['%d'%Nid]         = ptf['hazard_curves_MC_%d'%Nid]
        if RS_samp_scen>0:
           new_ens_RS = {}
           rangers=len(ptf['new_ensemble_RS'])
           hazard_curves_rs={}
           for Nid in range(rangers):
               new_ens_RS['%d'%Nid] = ptf['new_ensemble_RS'][Nid]
               hazard_curves_rs['%d'%Nid]         = ptf['hazard_curves_RS_%d'%Nid]

    '''
    if(status != 'no_message'):
        print(" --> Save preliminari alert message %s" % (ee['alert_message_file']))
        save_dict['alert_message_file'] = ee['alert_message_file']
        f = open(ee['alert_message_file'],'w')
        f.write(ee['tsunami_message_initial'])
        f.close()

        print(" --> Save preliminary message in json file format %s" % (save_dict['json_file']))
        json_dict = json.dumps(ee['message_dict'])
        open(save_dict['event_path'] + os.sep + save_dict['json_file'], 'w').write(json_dict)
    '''
    #pprint.pprint(json_dict)



    if (ptf != False and status != 'no_message'):
        # Hazard curves
        print(" --> Save hazard curves in file (%s-file format) %s.%s" % (format, save_dict['hazard_curves_original'],format))
        # print(" --> Save pois informations in file (%s-file format) %s.%s" % (format, save_dict['pois'],format))
        print(" --> Save event parameters in file (%s-file format) %s.%s" % (format, save_dict['event_parameters'],format))
        # print(" --> Save estimated alert levels at pois in file (%s-file format) %s.%s" % (format, save_dict['alert_levels'],format))

        if(format_hc == 'npy'):
            np.save(save_dict['event_path'] + os.sep + save_dict['hazard_curves_original'],   hazard_curves_original,    allow_pickle=True)
            if MC_samp_scen>0:
               for Nid in range(rangemc):
                   np.save(save_dict['event_path'] + os.sep + save_dict['hazard_curves_MC'] + '_' + '%d'%Nid,   hazard_curves_mc['%d'%Nid],    allow_pickle=True)
            if RS_samp_scen>0:
               for Nid in range(rangers):
                   np.save(save_dict['event_path'] + os.sep + save_dict['hazard_curves_RS'] + '_' + '%d'%Nid,   hazard_curves_rs['%d'%Nid],    allow_pickle=True)

            #np.save(save_dict['event_path'] + os.sep + save_dict['pois'],            pois,             allow_pickle=True)
            np.save(save_dict['event_path'] + os.sep + save_dict['event_parameters'],ee,               allow_pickle=True)
            # np.save(save_dict['event_path'] + os.sep + save_dict['alert_levels'],    alert_levels,     allow_pickle=True)

        if(format_hc == 'hdf5'):

            # Save hazard curve
            if OR_HC>0:
               h5file = save_dict['event_path'] + os.sep + save_dict['hazard_curves_original'] + '.hdf5'
               #hkl.dump(hazard_curves_original, h5file, mode='w')
               hf = h5py.File(save_dict['event_path'] + os.sep + save_dict['hazard_curves_original'] + '.hdf5', 'w')
               hf.create_dataset('hazard_curves_at_pois',            data = hazard_curves_original['hazard_curves_at_pois'])
               hf.create_dataset('hazard_curves_at_pois_mean',       data = hazard_curves_original['hazard_curves_at_pois_mean'])
               hf.create_dataset('hazard_curves_bs_at_pois',         data = hazard_curves_original['bs']['hazard_curves_bs_at_pois'])
               hf.create_dataset('hazard_curves_bs_at_pois_mean',    data = hazard_curves_original['bs']['hazard_curves_bs_at_pois_mean'])
               hf.create_dataset('generic_hazard_curve_threshold',   data = hazard_curves_original['generic_hazard_curve_threshold'])
               hf.create_dataset('original_hazard_curve_threshold',  data = hazard_curves_original['original_hazard_curve_threshold'])
               hf.create_dataset('tsunami_intensity_name',           data = hazard_curves_original['tsunami_intensity_name'])
               hf.create_dataset('hazard_curve_thresholds',          data = hazard_curves_original['hazard_curve_thresholds'])
               hf.create_dataset('runUp_amplification_factor',       data = hazard_curves_original['tsunami_intensity_runUp_amplification_factor'])
               if (probability_scenarios['nr_ps_scenarios'] > 0):
                   hf.create_dataset('hazard_curves_ps_at_pois',         data = hazard_curves_original['ps']['hazard_curves_ps_at_pois'])
                   hf.create_dataset('hazard_curves_ps_at_pois_mean',    data = hazard_curves_original['ps']['hazard_curves_ps_at_pois_mean'])
               else:
                   hf.create_dataset('hazard_curves_ps_at_pois',         data = np.array([0.0]))
                   hf.create_dataset('hazard_curves_ps_at_pois_mean',    data = np.array([0.0]))
               hf.close()
            
            h5file = save_dict['event_path'] + os.sep + save_dict['pois'] + '.hdf5'
            #hf = h5py.File(save_dict['event_path'] + os.sep + save_dict['pois'] + '.hdf5', 'w')
            #hf.create_dataset('pois',            data = pois[]	    
            hkl.dump(pois, h5file, mode='w')

            h5file = save_dict['event_path'] + os.sep + save_dict['event_parameters'] + '.hdf5'
            hkl.dump(ee, h5file, mode='w')

            #h5file = save_dict['event_path'] + os.sep + save_dict['alert_levels'] + '.hdf5'
            #hkl.dump(alert_levels, h5file, mode='w')

            if MC_samp_scen>0:
               for Nid in range(rangemc):
                   h5file = save_dict['event_path'] + os.sep + save_dict['hazard_curves_MC'] + '_' + '%d'%Nid + '.hdf5'
                   #hkl.dump(hazard_curves_mc['%d'%Nid], h5file, mode='w')
                   hf = h5py.File(save_dict['event_path'] + os.sep + save_dict['hazard_curves_MC'] + '_' + '%d'%Nid + '.hdf5', 'w')
                   hf.create_dataset('hazard_curves_at_pois', data=hazard_curves_mc['%d'%Nid]['hazard_curves_at_pois'])
                   hf.create_dataset('hazard_curves_at_pois_mean', data=hazard_curves_mc['%d'%Nid]['hazard_curves_at_pois_mean'])
                   hf.create_dataset('hazard_curves_bs_at_pois', data=hazard_curves_mc['%d'%Nid]['bs']['hazard_curves_bs_at_pois'])
                   hf.create_dataset('hazard_curves_bs_at_pois_mean', data=hazard_curves_mc['%d'%Nid]['bs']['hazard_curves_bs_at_pois_mean'])
                   hf.create_dataset('tsunami_intensity_name', data=hazard_curves_mc['%d'%Nid]['tsunami_intensity_name'])
                   hf.create_dataset('Intensity_measure_all_bs', data=hazard_curves_mc['%d'%Nid]['bs']['Intensity_measure_all_bs'])
                   if (new_ens_MC['%d'%Nid]['nr_ps_scenarios'] > 0):
                       hf.create_dataset('hazard_curves_ps_at_pois', data=hazard_curves_mc['%d'%Nid]['ps']['hazard_curves_ps_at_pois'])
                       hf.create_dataset('hazard_curves_ps_at_pois_mean', data=hazard_curves_mc['%d'%Nid]['ps']['hazard_curves_ps_at_pois_mean'])
                   else:
                       hf.create_dataset('hazard_curves_ps_at_pois', data=np.array([0.0]))
                       hf.create_dataset('hazard_curves_ps_at_pois_mean', data=np.array([0.0]))
                   hf.close()

            if RS_samp_scen>0:
               for Nid in range(rangers):
                   h5file = save_dict['event_path'] + os.sep + save_dict['hazard_curves_RS'] + '_' + '%d'%Nid + '_' + '%d'%inter +'.hdf5'
                   hf = h5py.File(save_dict['event_path'] + os.sep + save_dict['hazard_curves_RS'] + '_' + '%d'%Nid + '_' + '%d'%inter +'.hdf5', 'w')
                   hf.create_dataset('hazard_curves_at_pois', data=hazard_curves_rs['%d'%Nid]['hazard_curves_at_pois'])
                   hf.create_dataset('hazard_curves_at_pois_mean', data=hazard_curves_rs['%d'%Nid]['hazard_curves_at_pois_mean'])
                   hf.create_dataset('hazard_curves_bs_at_pois', data=hazard_curves_rs['%d'%Nid]['bs']['hazard_curves_bs_at_pois'])
                   hf.create_dataset('hazard_curves_bs_at_pois_mean', data=hazard_curves_rs['%d'%Nid]['bs']['hazard_curves_bs_at_pois_mean'])
                   hf.create_dataset('tsunami_intensity_name', data=hazard_curves_rs['%d'%Nid]['tsunami_intensity_name'])
                   hf.create_dataset('Intensity_measure_all_bs', data=hazard_curves_rs['%d'%Nid]['bs']['Intensity_measure_all_bs'])
                   if (new_ens_RS['%d'%Nid]['nr_ps_scenarios'] > 0):
                       hf.create_dataset('hazard_curves_ps_at_pois', data=hazard_curves_rs['%d'%Nid]['ps']['hazard_curves_ps_at_pois'])
                       hf.create_dataset('hazard_curves_ps_at_pois_mean', data=hazard_curves_rs['%d'%Nid]['ps']['hazard_curves_ps_at_pois_mean'])
                   else:
                       hf.create_dataset('hazard_curves_ps_at_pois', data=np.array([0.0]))
                       hf.create_dataset('hazard_curves_ps_at_pois_mean', data=np.array([0.0]))
                   hf.close()


    return save_dict

def save_ptf_out(**kwargs):


    Config                = kwargs.get('cfg', None)
    args                  = kwargs.get('args', None)
    ee                    = kwargs.get('event_parameters', None)
    ptf                   = kwargs.get('ptf', None)
    status                = kwargs.get('status', None)
    sim_files             = kwargs.get('sim_files', None)

    if args.save_format != None:
        format = args.save_format
    else:
        format    = Config.get('save_ptf','save_format')

    save_dict = define_save_path(cfg = Config,args = args, event = ee, sim_files = sim_files)

    save_dict = define_file_names(cfg = Config,args = args, event = ee, dictionary=save_dict)

    #######################
    if(ptf != False):

        h5file = save_dict['event_path'] + os.sep + 'ptf_out' + '.hdf5'
        hkl.dump(ptf, h5file, mode='w')
        #print(h5file)        
 
    return save_dict

def save_ptf_out_int(**kwargs):


    Config                = kwargs.get('cfg', None)
    args                  = kwargs.get('args', None)
    ee                    = kwargs.get('event_parameters', None)
    ptf                   = kwargs.get('ptf', None)
    status                = kwargs.get('status', None)
    sim_files             = kwargs.get('sim_files', None)
    list_tmp_scen         = kwargs.get('list_tmp_scen', None)

    inter = list_tmp_scen

    if args.save_format != None:
        format = args.save_format
    else:
        format    = Config.get('save_ptf','save_format')

    save_dict = define_save_path(cfg = Config,args = args, event = ee, sim_files = sim_files)

    save_dict = define_file_names(cfg = Config,args = args, event = ee, dictionary=save_dict)

    #######################
    if(ptf != False):

        h5file = save_dict['event_path'] + os.sep + 'ptf_out_' + '%d'%inter + '.hdf5'
        hkl.dump(ptf, h5file, mode='w')
        #print(h5file)        

    return save_dict


def load_ptf_out(**kwargs):
    Config                = kwargs.get('cfg', None)
    args                  = kwargs.get('args', None)
    ee                    = kwargs.get('event_parameters', None)
    status                = kwargs.get('status', None)
    sim_files             = kwargs.get('sim_files', None)

    if args.save_format != None:
        format = args.save_format
    else:
        format    = Config.get('save_ptf','save_format')

    save_dict = define_save_path(cfg = Config,args = args, event = ee, sim_files = sim_files)

    save_dict = define_file_names(cfg = Config,args = args, event = ee, dictionary=save_dict)

    #######################
    
    ptf_out={}
    h5file = save_dict['event_path'] + os.sep + 'ptf_out' + '.hdf5'
    ptf_out=hkl.load(h5file)

    return ptf_out

