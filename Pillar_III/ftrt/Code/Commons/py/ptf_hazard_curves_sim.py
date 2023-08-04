
import h5py
import scipy
import numpy as np
from ismember          import ismember


def compute_hazard_curves_sim(**kwargs):

    Config                = kwargs.get('cfg', None)
    ee                    = kwargs.get('event_parameters', None)
    args                  = kwargs.get('args', None)
    LongTermInfo          = kwargs.get('LongTermInfo', None)
    probability_scenarios = kwargs.get('probability_scenarios', None)
    Scenarios_BS          = kwargs.get('Scenarios_BS', None)
    Scenarios_PS          = kwargs.get('Scenarios_PS', None)
    pois                  = kwargs.get('pois', None)
    h_curve_files         = kwargs.get('h_curve_files', None)
    short_term            = kwargs.get('short_term_probability', None)
    data                  = kwargs.get('data', None)
    list_tmp_scen         = kwargs.get('list_tmp_scen', None)

    print(len(probability_scenarios['real_par_scenarios_bs']))

    # This may be moved at the biginn for CAT standard configuration
    hazards = define_thresholds(cfg = Config, LongTermInfo = LongTermInfo, args = args, pois = pois)
    short_term_BS = True
    short_term_PS = False

    if (probability_scenarios['PScomputedYN'] == True and short_term_PS == True and len(probability_scenarios['par_scenarios_ps'][:,0]) > 0):

        print(" --> Hazard Curves for PS scenarios")
        hazards_ps = hazard_curves_ps(cfg               = Config,
                                   args              = args,
                                   hazards           = hazards,
                                   pois              = pois,
                                   probability_scene = probability_scenarios,
                                   Scenarios_PS      = Scenarios_PS,
                                   h_curve_files     = h_curve_files)


    #if (probability_scenarios['BScomputedYN'] == True and short_term['BS_computed_YN'] == True and len(probability_scenarios['real_par_scenarios_bs']) > 0):
    if (len(probability_scenarios['real_par_scenarios_bs']) > 0):

        print(" --> Hazard Curves for BS scenarios")

        hazards_bs = hazard_curves_bs(cfg               = Config,
                                   args              = args,
                                   hazards           = hazards,
                                   pois              = pois,
                                   probability_scene = probability_scenarios,
                                   Scenarios_BS      = Scenarios_BS,
                                   #data              = data,
                                   list_tmp_scen     = list_tmp_scen,
                                   h_curve_files     = h_curve_files)



    #print("--------------------------->", short_term['PS_computed_YN'])
    if(short_term_PS == True and short_term_BS == True):
        hazards['hazard_curves_at_pois']      = hazards_bs['hazard_curves_bs_at_pois'] + \
                                                hazards_ps['hazard_curves_ps_at_pois']
        hazards['hazard_curves_at_pois_mean'] = hazards_bs['hazard_curves_bs_at_pois_mean'] + \
                                                hazards_ps['hazard_curves_ps_at_pois_mean']
        hazards['ps'] = hazards_ps
        hazards['bs'] = hazards_bs
        if(probability_scenarios['real_best_scenarios']['max_ValBS'] > probability_scenarios['real_best_scenarios']['max_ValPS']):
            best_idx       = probability_scenarios['real_best_scenarios']['max_idxBS']
            best_intensity = hazards['Intensity_measure_all_bs'][best_idx,:]
        else:
            best_idx = probability_scenarios['real_best_scenarios']['max_idxPS']
            best_intensity = hazards['Intensity_measure_all_ps'][best_idx,:]


    elif(short_term_PS == True and short_term_BS == False):
        hazards['hazard_curves_at_pois']      = hazards_ps['hazard_curves_ps_at_pois']
        hazards['hazard_curves_at_pois_mean'] = hazards_ps['hazard_curves_ps_at_pois_mean']
        hazards['ps'] = hazards_ps
        hazards['bs'] = False
        best_idx = probability_scenarios['real_best_scenarios']['max_idxPS']
        best_intensity = hazards['Intensity_measure_all_ps'][best_idx,:]

    elif(short_term_PS == False and short_term_BS == True):
        hazards['hazard_curves_at_pois']      = hazards_bs['hazard_curves_bs_at_pois']
        hazards['hazard_curves_at_pois_mean'] = hazards_bs['hazard_curves_bs_at_pois_mean']
        hazards['ps'] = False
        hazards['bs'] = hazards_bs
        best_idx = probability_scenarios['real_best_scenarios']['max_idxBS']
        best_intensity = hazards['Intensity_measure_all_bs'][best_idx,:]
    else:
        print("WARNINGS!!!! ")
        print("Use Decision Matrix: info for low mag, wathc for larger mag")

        hazards['use_matrix'] = True
        return hazards


    # Merge harzard ps and bs
    """
    if(probability_scenarios['nr_ps_scenarios'] > 0 and short_term['PS_computed_YN'] == True):
        hazards['hazard_curves_at_pois']      = hazards_bs['hazard_curves_bs_at_pois'] + \
                                                hazards_ps['hazard_curves_ps_at_pois']
        hazards['hazard_curves_at_pois_mean'] = hazards_bs['hazard_curves_bs_at_pois_mean'] + \
                                                hazards_ps['hazard_curves_ps_at_pois_mean']
        hazards['ps'] = hazards_ps
        hazards['bs'] = hazards_bs

    else:
        hazards['hazard_curves_at_pois']      = hazards_bs['hazard_curves_bs_at_pois']
        hazards['hazard_curves_at_pois_mean'] = hazards_bs['hazard_curves_bs_at_pois_mean']
        hazards['bs'] = hazards_bs


    if(probability_scenarios['best_scenarios']['max_ValBS'] > probability_scenarios['best_scenarios']['max_ValPS']):
        best_idx       = probability_scenarios['best_scenarios']['max_idxBS']
        best_intensity = hazards['Intensity_measure_all_bs'][best_idx,:]
    else:
        best_idx = probability_scenarios['best_scenarios']['max_idxPS']
        best_intensity = hazards['Intensity_measure_all_ps'][best_idx,:]
    """
    hazards['use_matrix']      = False
    hazards['best_Intensity']  = best_intensity


    return hazards

def hazard_curves_ps(**kwargs):

    Config                = kwargs.get('cfg', None)
    args                  = kwargs.get('args', None)
    probability_scenarios = kwargs.get('probability_scene', None)
    Scenarios_PS          = kwargs.get('Scenarios_PS', None)
    hazards               = kwargs.get('hazards', None)
    pois                  = kwargs.get('pois', None)
    h_curve_files         = kwargs.get('h_curve_files', None)

    hazard_curves_pois    = hazards['pois_hazard_curves_PS']
    hazard_curves_pois_m  = np.zeros(pois['nr_selected_pois'])

    sigma                 = float(Config.get('Settings','hazard_curve_sigma'))
    type_of_measure       = args.intensity_measure

    vec = np.array([100000000,100000,100,1,0.0100])#,1.0000e-04,1.0000e-06])

    relevant_scenarios_ps    = probability_scenarios['relevant_scenarios_ps']
    relevant_idx_nr          = []
    relevant_idx_nr_max_zero = [] #Nr of idx larger than 0

    Intensity_measure_all = np.zeros((probability_scenarios['nr_ps_scenarios'], len(hazard_curves_pois_m)))

    for k in range(len(relevant_scenarios_ps)):

        print("     --> Region %d " % (relevant_scenarios_ps[k]))

        nr                     = relevant_scenarios_ps[k]
        scene_matrix           = np.array(Scenarios_PS[nr]['Parameters'])
        scene_matrix[:,3]      = np.array(Scenarios_PS[nr]['modelVal'][:])
        scene_matrix[:,4]      = np.transpose(np.array(Scenarios_PS[nr]['SlipDistribution']))
        scene_matrix           = np.transpose(scene_matrix)
        convolved_scenarios_ps = vec.dot(scene_matrix)
        relevant_file_scenario = int(nr) -1

        isel                   = np.where(probability_scenarios['par_scenarios_ps'][:,0]==nr)
        #print("....................", nr, isel)
        arra                   = probability_scenarios['par_scenarios_ps'][isel]
        par_matrix             = np.transpose(arra[:,1:6])  #8
        convolved_par_ps       = vec.dot(par_matrix)

        Iloc,nr_scenarios      = ismember(convolved_par_ps,convolved_scenarios_ps)
        idx_true_scenarios     = np.where([nr_scenarios>0])[1]

        probsel                = probability_scenarios['ProbScenPS'][isel[0]]

        print("Lenght probsel",len(probability_scenarios['ProbScenPS'][isel[0]]))

        """
        questa parte va ricontrollata, ovvero:
        da /data/TSUMAPS1.1/MatPTF_ProcessedFiles_simplified/glVal_*mat
        a /data/pyPTF/hazard_curves/glVal_*hdf5, oppure già che ci siamo a netcdf
        """
        #return

        # Open and read hdf5-curves file
        infile = h_curve_files['gl_ps'][relevant_file_scenario]

        if(nr_scenarios.size == 0):
            continue
        #print("----->", infile)
        #fb hdf5 to npy
        with h5py.File(infile, "r") as f:
            a_group_key = list(f.keys())[0]
            data = np.array(f.get(a_group_key))

        #npy
        #data = np.load(infile)

        for ip in range(0,pois['nr_selected_pois']):

            #InputIntensityMeasure = relevant_data
            # Questo potrebbe essere messo fuori che è sempre identico
            hazard_curve_threshold = hazards['original_hazard_curve_threshold'].reshape(1, len(hazards['original_hazard_curve_threshold']))


            try:
                InputIntensityMeasure  = np.take(data[ip], nr_scenarios) #InputIntensityMeasure, itmpInd = InputIntensityMeasure > 0; (tutti zero e uni)
            except:
                InputIntensityMeasure  = np.take(data[ip], nr_scenarios)

            itmpInd                = np.array(InputIntensityMeasure > 0)
            
            try:
                prob_tmp_scenarios     = probsel[itmpInd]
            except:
                print(ip, nr, np.shape(probsel), np.shape(itmpInd))
                prob_tmp_scenarios     = probsel[itmpInd]


            mu                     = np.log(InputIntensityMeasure[itmpInd])
            mu                     = mu.reshape(len(mu), 1)

            cond_hazard_curve_tmp       = 1 - scipy.stats.lognorm.cdf(hazard_curve_threshold, sigma, scale=np.exp(mu)).transpose()
            cond_hazard_curve_tmp_mean  = np.exp(mu + 0.5*sigma**2)

            prob_tmp                    = np.matlib.repmat(prob_tmp_scenarios, 1, np.size(hazard_curve_threshold)).reshape(np.shape(cond_hazard_curve_tmp))

            hazard_curves_pois[ip,:]    = hazard_curves_pois[ip,:] + np.sum(prob_tmp*cond_hazard_curve_tmp, axis=1)
            hazard_curves_pois_m[ip]    = hazard_curves_pois_m[ip] + np.sum(prob_tmp_scenarios*cond_hazard_curve_tmp_mean.transpose()[0])

            Intensity_measure_all[isel,ip] = InputIntensityMeasure

    hazards['hazard_curves_ps_at_pois']      = hazard_curves_pois
    hazards['hazard_curves_ps_at_pois_mean'] = hazard_curves_pois_m
    hazards['Intensity_measure_all_ps']      = Intensity_measure_all

    return hazards

def select_hazard_curves_for_bs_k_scenarios(**kwargs):

    Config                = kwargs.get('cfg', None)
    args                  = kwargs.get('args', None)
    probability_scenarios = kwargs.get('probability_scene', None)
    Scenarios_BS          = kwargs.get('Scenarios_BS', None)
    hazards               = kwargs.get('hazards', None)
    pois                  = kwargs.get('pois', None)
    h_curve_files         = kwargs.get('h_curve_files', None)
    sel_scen              = kwargs.get('list_tmp_scen', None)

    #data                  = kwargs.get('data', None)

    nr_cpu_allowed        = float(Config.get('Settings','nr_cpu_max'))
    sigma                 = float(Config.get('Settings','hazard_curve_sigma'))
    type_of_measure       = args.intensity_measure

    hazard_curves_pois    = hazards['pois_hazard_curves_BS']
    hazard_curves_pois_m  = np.zeros(pois['nr_selected_pois'])
    nr_pois               = pois['nr_selected_pois']
    pois_hc               = len(hazards['original_hazard_curve_threshold'])
    hazard_curves_pois    = np.zeros((nr_pois, pois_hc))

    #Intensity_measure_all    = np.zeros((probability_scenarios['nr_bs_scenarios'], len(hazard_curves_pois_m)))
    probsel_tmp                  = probability_scenarios['RealProbScenBS']
    tot_prob                     = np.sum(probsel_tmp)
    probsel                      = probsel_tmp/tot_prob
    
    # Open and read hdf5-curves file
    infile = h_curve_files['gl_bs']
    data = np.transpose(infile)
    
    hazard_curve_threshold = hazards['original_hazard_curve_threshold'].reshape(1, len(hazards['original_hazard_curve_threshold']))
    #nr_scenarios = isel[0]
    InputIntensityMeasure = data#[:, nr_scenarios]
    ItmpInd               = data> 0 #[:, nr_scenarios] > 0
    Intensity_measure_all = np.zeros((len(sel_scen), len(hazard_curves_pois_m)))

    S = np.ma.masked_less_equal(InputIntensityMeasure, 0)
    #S = np.ma.masked_values(S, -9999.0)
    #S = M

    #print(secondary_loop_idxs,'\n\n')
    u = np.ones(pois['nr_selected_pois']) * -1
 
    for ip in range(0,pois['nr_selected_pois']):

        prob_tmp_scenarios          = probsel[ItmpInd[ip,:]]

        #mu                          = np.log(InputIntensityMeasure[ip,:][ItmpInd[ip,:]])
        #mu                          = mu.reshape(len(mu), 1)
        ss                          = S[ip,:][~S[ip,:].mask].data
        MU                          = np.log(S[ip,:][~S[ip,:].mask].data)
        #print(MU)
        mu                          = MU.reshape(len(MU[0]), 1)
        cond_hazard_curve_tmp       = 1 - scipy.stats.lognorm.cdf(hazard_curve_threshold, sigma, scale=np.exp(mu)).transpose()
        cond_hazard_curve_tmp_mean  = np.exp(mu + 0.5*sigma**2)
        prob_tmp                    = np.matlib.repmat(prob_tmp_scenarios, np.size(hazard_curve_threshold), 1).reshape(np.shape(cond_hazard_curve_tmp))

        hazard_curves_pois[ip,:]    = np.sum(prob_tmp           * cond_hazard_curve_tmp, axis=1)
        hazard_curves_pois_m[ip]    = np.sum(prob_tmp_scenarios * cond_hazard_curve_tmp_mean.transpose()[0])

        Intensity_measure_all[:,ip] = InputIntensityMeasure[ip,:]

    return hazard_curves_pois, hazard_curves_pois_m, Intensity_measure_all



    """
    for ip in range(0,pois['nr_selected_pois']):

        prob_tmp_scenarios          = probsel[ItmpInd[ip,:]]

        #mu                          = np.log(InputIntensityMeasure[ip,:][ItmpInd[ip,:]])
        #mu                          = mu.reshape(len(mu), 1)
        ss = S[ip,:][~S[ip,:].mask].data
        MU                          = np.log(S[ip,:][~S[ip,:].mask].data)

        mu                          = MU.reshape(len(MU), 1)

        cond_hazard_curve_tmp       = 1 - scipy.stats.lognorm.cdf(hazard_curve_threshold, sigma, scale=np.exp(mu)).transpose()
        cond_hazard_curve_tmp_mean  = np.exp(mu + 0.5*sigma**2)

        prob_tmp                    = np.matlib.repmat(prob_tmp_scenarios, 1, np.size(hazard_curve_threshold)).reshape(np.shape(cond_hazard_curve_tmp))

        hazard_curves_pois[ip,:]    = np.sum(prob_tmp           * cond_hazard_curve_tmp, axis=1)
        hazard_curves_pois_m[ip]    = np.sum(prob_tmp_scenarios * cond_hazard_curve_tmp_mean.transpose()[0])

        Intensity_measure_all[isel,ip] = InputIntensityMeasure[ip,:]
    """

def merge_hazard_curves(r, n):

    hc_list = []
    for i in range(len(r)):
        hc_list.append(r[i][n])
    out = np.add.reduce(hc_list)

    return out

def hazard_curves_bs(**kwargs):

    Config                = kwargs.get('cfg', None)
    args                  = kwargs.get('args', None)
    probability_scenarios = kwargs.get('probability_scene', None)
    Scenarios_BS          = kwargs.get('Scenarios_BS', None)
    hazards               = kwargs.get('hazards', None)
    pois                  = kwargs.get('pois', None)
    h_curve_files         = kwargs.get('h_curve_files', None)
    data                  = kwargs.get('data', None)
    list_tmp_scen         = kwargs.get('list_tmp_scen', None)

    #hazard_curves_pois    = hazards['pois_hazard_curves_BS']
    #hazard_curves_pois_m  = np.zeros(pois['nr_selected_pois'])

    sigma                 = float(Config.get('Settings','hazard_curve_sigma'))
    type_of_measure       = args.intensity_measure

    result_ids = []

    result_ids.append(select_hazard_curves_for_bs_k_scenarios(  cfg               = Config,
                                                                args              = args,
                                                                hazards           = hazards,
                                                                pois              = pois,
                                                                data              = data,
                                                                probability_scene = probability_scenarios,
                                                                Scenarios_BS      = Scenarios_BS,
                                                                list_tmp_scen     = list_tmp_scen,
                                                                h_curve_files     = h_curve_files))

    results = result_ids

    HC_P = merge_hazard_curves(results, 0)
    HC_M = merge_hazard_curves(results, 1)
    IM_P = merge_hazard_curves(results, 2)

    hazards['hazard_curves_bs_at_pois']      = HC_P
    hazards['hazard_curves_bs_at_pois_mean'] = HC_M
    hazards['Intensity_measure_all_bs']      = IM_P

    return hazards


def define_thresholds(**kwargs):

    Config          = kwargs.get('cfg', None)
    LongTermInfo    = kwargs.get('LongTermInfo', None)
    args            = kwargs.get('args', None)
    pois            = kwargs.get('pois', None)

    #type_of_measure = Config.get('Settings', 'selected_intensity_measure')
    type_of_measure = args.intensity_measure
    run_up_yn       = args.compute_runUp

    conf = dict()


    generic_hazard_curve_threshold = compute_generic_hazard_curve_threshold(curve = 'standard')


    conf['generic_hazard_curve_threshold']  = generic_hazard_curve_threshold
    conf['original_hazard_curve_threshold'] = generic_hazard_curve_threshold

    if(type_of_measure == 'gl'):

        if(run_up_yn == 'y'):
            conf['tsunami_intensity_name'] = 'Maximum run-up from GL (m)'
            conf['tsunami_intensity_runUp_amplification_factor'] = 2
        else:
            conf['tsunami_intensity_name'] = 'GL - Green\'s law (m)'
            conf['tsunami_intensity_runUp_amplification_factor'] = 1

    if(type_of_measure == 'os'):

        conf['tsunami_intensity_name']          = 'HMax - Off-Shore wave height (m)'
        conf['tsunami_intensity_runUp_amplification_factor'] = 1

    if(type_of_measure == 'af'):

        if(run_up_yn == 'y'):
            conf['tsunami_intensity_name'] = 'Maximum run-up from MIH (m)'
            conf['tsunami_intensity_runUp_amplification_factor'] = 3
        else:
            conf['tsunami_intensity_name'] = 'MIH - Maximum Inundatiuon Height (m)'
            conf['tsunami_intensity_runUp_amplification_factor'] = 1

        conf['original_hazard_curve_threshold'] = LongTermInfo['LookupTableConditionalHazardCurves']['HazardCurveThresholds']

    conf['hazard_curve_thresholds'] = conf['original_hazard_curve_threshold'] * conf['tsunami_intensity_runUp_amplification_factor']
    conf['nr_thresholds']           = len(conf['hazard_curve_thresholds'])

    # Initizalizate matrix for HazardCurves.hc_poiBS, with dimensiono(nr_pois, len(original_hazard_curve_threshold))
    nr_pois = pois['nr_selected_pois']
    pois_hc = len(conf['original_hazard_curve_threshold'])
    conf['pois_hazard_curves_BS'] = np.zeros((nr_pois, pois_hc))
    conf['pois_hazard_curves_PS'] = np.zeros((nr_pois, pois_hc))

    return conf

def compute_generic_hazard_curve_threshold(**kwargs):

    ### Here we set the intensity thresholds in meters

    curve = kwargs.get('curve', None)

    if(curve == 'standard'):
        # Values on meters
        out=np.array([1.0e-05,0.0001,0.0010,0.0100,0.0200,0.0300,0.0400,0.0500,\
                        0.1000,0.1500,0.2000,0.2500,0.3000,0.3500,0.4000,0.4500,\
                        0.5000,0.6000,0.7000,0.8000,0.9000,1.0000,1.1100,1.3140,\
                        1.5370,1.7820,2.0500,2.3440,2.6670,3.0210,3.4090,3.8340,\
                        4.3010,4.8120,5.3730,5.9870,6.6620,7.4010,8.2110,9.1000,\
                        10.0740,11.1430,12.3140,13.5990,15.0070,16.5520,18.2450,\
                        20.1020,22.1380,24.3700,26.8180,29.5020,32.4440,35.6710,\
                        39.2090,43.0880,47.3420,52.0060,57.1190,62.7270,68.8750,\
                        75.6160,83.0080,91.1130,100.0000])


    if(curve == 'short'):
        tmp = -10**(np.arange(-2,1,0.1))
        ext = np.array([1e-5, 1e-4, 1e-3, 25.0, 50.0, 100.0])
        out = np.insert(ext, 3, -10**(np.arange(-2,1,0.1)) )

    return out
