
import numpy as np

def compute_ensemble_sampling(**kwargs):


    Config         = kwargs.get('cfg', None)
    ee             = kwargs.get('event_parameters', None)
    args           = kwargs.get('args', None)
    LongTermInfo   = kwargs.get('LongTermInfo', None)
    PSBarInfo      = kwargs.get('PSBarInfo', None)
    lambda_bsps    = kwargs.get('lambda_bsps', None)
    pre_selection  = kwargs.get('pre_selection', None)
    short_term     = kwargs.get('short_term', None)
    regions        = kwargs.get('regions', None)
    Scenarios_PS   = kwargs.get('Scenarios_PS', None)
    probability_scenarios  = kwargs.get('proba_scenarios', None)

    TotProbBS_all = np.sum(probability_scenarios['ProbScenBS'])
    TotProbPS_all = np.sum(probability_scenarios['ProbScenPS'])
    #Nsamp=int(Config.get('Settings','Sampling'))
    #Nsamp=[10,50,100,500,1000,5000,10000,15000,20000]
    Nsamp=[]#np.ones(100)*50000
    #Nsamp=[1000,20000]
    #np.arange(1,20000,100)
    sampled_ensemble = {}

    Nid=0
    for N in Nsamp:
        NBS= int(TotProbBS_all*N)
        print('Nbr sampling', NBS)
        NPS= N-NBS
        sampled_ensemble[Nid] = dict()
        sampled_ensemble[Nid] = set_if_compute_scenarios(cfg           = Config,
                                                short_term    = short_term)
        #sampled_ensemble[Nid] = dict()
        #prob_vec = probability_scenarios['prob_scenarios_ps_fact']['ProbScenBS'][]
        prob_len = len(probability_scenarios['ProbScenBS'])
        #print(prob_len)
        intervals_ensemble = np.zeros(prob_len)
        prob_cum = 0
        for i in range(prob_len):
            prob_cum=prob_cum+probability_scenarios['ProbScenBS'][i]
            intervals_ensemble[i]= prob_cum

        ### proba, index, parameters
        sampled_ensemble[Nid]['prob_scenarios_bs_fact'] = np.zeros( (NBS,  5) )
        sampled_ensemble[Nid]['prob_scenarios_bs'] = np.zeros( (NBS) )
        sampled_ensemble[Nid]['par_scenarios_bs'] = np.zeros(  (NBS, 11) )
        sampled_ensemble[Nid]['prob_scenarios_ps_fact'] = np.zeros( (NBS,  5) )
        sampled_ensemble[Nid]['prob_scenarios_ps'] = np.zeros( (NBS) )
        sampled_ensemble[Nid]['par_scenarios_ps'] = np.zeros(  (NBS,  7) )
        sampled_ensemble[Nid]['iscenbs']=np.zeros(NBS)
        sampled_ensemble[Nid]['iscenps']=np.zeros(NBS)

        sampled_ensemble = bs_probability_scenarios(cfg              = Config,
                                                     short_term       = short_term,
                                                     pre_selection    = pre_selection,
                                                     regions_files    = regions,
                                                     prob_scenes      = probability_scenarios,
                                                     samp_ens         = sampled_ensemble,
                                                     Discretizations  = LongTermInfo['Discretizations'],
						     NBS	      = NBS,
                                                     Nid	      = Nid,
                                                     intervals_ensemble = intervals_ensemble)

        sampled_ensemble = ps_probability_scenarios(cfg              = Config,
                                                         PSBarInfo        = PSBarInfo,
                                                         short_term       = short_term,
                                                         pre_selection    = pre_selection,
                                                         prob_scenes      = probability_scenarios,
                                                         samp_ens         = sampled_ensemble,
                                                         region_ps        = LongTermInfo['region_listPs'],
                                                         Model_Weights    = LongTermInfo['Model_Weights'],
                                                         Scenarios_PS     = Scenarios_PS,
                                                         ps1_magnitude    = LongTermInfo['Discretizations']['PS-1_Magnitude'],
                                                         lambda_bsps      = lambda_bsps,
							 Nid              = Nid)
    
        if(sampled_ensemble[Nid] == False):
            return False

        # check on the nr scenarios computed into the two section. Should be identical
        check_bs = 'OK'
        check_ps = 'OK'

        if(sampled_ensemble[Nid]['nr_bs_scenarios'] != short_term['Total_BS_Scenarios']):
            check_bs = 'WARNING'
        if(sampled_ensemble[Nid]['nr_ps_scenarios'] != short_term['Total_PS_Scenarios']):
            check_ps = 'WARNING'

        print(' --> Check Nr Bs scenarios: %7d  <--> %7d --> %s' % (sampled_ensemble[Nid]['nr_bs_scenarios'], short_term['Total_BS_Scenarios'], check_bs))
        print(' --> Check Nr Ps scenarios: %7d  <--> %7d --> %s' % (sampled_ensemble[Nid]['nr_ps_scenarios'], short_term['Total_PS_Scenarios'], check_ps))
        
        # Re-Normalize scenarios (to manage events outside nSigma, BS for large events, ...)
        ProbScenBS        = sampled_ensemble[Nid]['prob_scenarios_bs_fact'].prod(axis=1)
        ProbScenPS        = sampled_ensemble[Nid]['prob_scenarios_ps_fact'].prod(axis=1)
        #print(len(ProbScenBS))
        TotProbBS_preNorm = np.sum(ProbScenBS)
        TotProbPS_preNorm = np.sum(ProbScenPS)
        TotProb_preNorm   = TotProbBS_preNorm + TotProbPS_preNorm

        # No scenarios bs or ps possible
        if(TotProb_preNorm == 0):
            return False
        elif(TotProb_preNorm != 0):
            ProbScenBS = ProbScenBS / TotProb_preNorm
            ProbScenPS = ProbScenPS / TotProb_preNorm
            print(' --> Total Bs scenarios probability pre-renormalization: %.5f' % TotProbBS_preNorm)
            print(' --> Total Ps scenarios probability pre-renormalization: %.5f' % TotProbPS_preNorm)
            print('     --> Total Bs and Ps probabilty renormalized to 1')

        Totnumber=len(ProbScenBS)
        ProbScenBS = np.ones(Totnumber)
        #print(Totnumber)
        sampled_ensemble[Nid]['ProbScenBS'] = np.ones(Totnumber)*1./Totnumber
        #sampled_ensemble[Nid]['ProbScenPS'] = np.ones(Totnumber)*1./Totnumber
        #print('sum before',np.sum(sampled_ensemble[Nid]['ProbScenBS'][:]))
        ##sampled_ensemble[Nid]['ProbScenBS'] = ProbScenBS
        ##sampled_ensemble[Nid]['ProbScenPS'] = ProbScenPS

        ######### Duplication of scenarios beginning ##########
        ##
        sample_unique_bs, test, counts_bs = np.unique(sampled_ensemble[Nid]['iscenbs'],return_index=True,return_counts=True) 
        #np.concatenate((sampled_ensemble[Nid]['iscenbs'][:],sampled_ensemble[Nid]['par_scenarios_bs']), axis=0)
        unique_par = sampled_ensemble[Nid]['par_scenarios_bs'][test,:]
        unique_fact = sampled_ensemble[Nid]['prob_scenarios_bs_fact'][test,:]
        unique_prob = sampled_ensemble[Nid]['ProbScenBS'][test]
        unique_name = sampled_ensemble[Nid]['iscenbs'][test]
        #sample_unique_bs, counts_bs = np.unique(sampled_ensemble[Nid]['iscenbs'],return_counts=True)
        ProbScenBS = np.ones(len(unique_prob))*1./Totnumber
        for itmp in range(len(unique_prob)):
           #iscenbs=sampled_ensemble[Nid]['iscenbs'][itmp]
           iscenbs=unique_name[itmp]
           indexbs=np.where(sample_unique_bs == iscenbs)
           ProbScenBS[itmp]=unique_prob[itmp]*counts_bs[indexbs]
        
        TotProbBS = np.sum(ProbScenBS)
        ###TotProbPS = np.sum(ProbScenPS)
        ##ProbScenBS = ProbScenBS / (TotProbBS)
        ###ProbScenPS = ProbScenPS / (TotProbBS+TotProbPS)
        ##
        ##sampled_ensemble[Nid]['ProbScenBS'] = ProbScenBS
        ##sampled_ensemble[Nid]['ProbScenPS'] = ProbScenPS

        sampled_ensemble[Nid]['par_scenarios_bs'] = unique_par
        sampled_ensemble[Nid]['prob_scenarios_bs_fact'] = unique_fact
        sampled_ensemble[Nid]['ProbScenBS'] = ProbScenBS

        sampled_ensemble[Nid]['relevant_scenarios_bs'] = np.unique(sampled_ensemble[Nid]['par_scenarios_bs'][:,0])
        sampled_ensemble[Nid]['relevant_scenarios_ps'] = np.unique(sampled_ensemble[Nid]['par_scenarios_ps'][:,0])

        print(' --> Relevant Scenarios BS : ', sampled_ensemble[Nid]['relevant_scenarios_bs'])
        print(' --> Relevant Scenarios PS : ', sampled_ensemble[Nid]['relevant_scenarios_ps'])

        try:
            max_idxBS = np.argmax(ProbScenBS)
        except:
            max_idxBS = -1
        try:
            max_ValBS = ProbScenBS[max_idxBS]
        except:
            max_ValBS = 0
        try:
            max_idxPS = np.argmax(ProbScenPS)
        except:
            max_idxPS = -1
        try:
            max_ValPS = ProbScenPS[max_idxPS]
        except:
            max_ValPS = 0

            max_ValPS = 0

            max_ValPS = 0

        sampled_ensemble[Nid]['best_scenarios'] = {'max_idxBS':max_idxBS, 'max_idxPS':max_idxPS, 'max_ValBS':max_ValBS, 'max_ValPS':max_ValPS}

        print('     --> Best Bs scenario Idx and Value: %6d    %.5e' % (max_idxBS, max_ValBS))
        print('     --> Best Ps scenario Idx and Value: %6d    %.5e' % (max_idxPS, max_ValPS))
        
        Nid=Nid+1

    return sampled_ensemble

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if (value-array[idx])>0:
        idx=idx+1
    return idx,array[idx]

def set_if_compute_scenarios(**kwargs):

    short_term     = kwargs.get('short_term', None)
    Config         = kwargs.get('cfg', None)

    neg_prob       = float(Config.get('Settings','negligible_probability'))

    out = dict()

    # Some inizialization
    out['nr_ps_scenarios'] = 0
    out['nr_bs_scenarios'] = 0

    BScomputedYN   = False
    PScomputedYN   = False

    tmpbs      = (short_term['magnitude_probability'] * short_term['RatioBSonTot']).sum()
    tmpps      = (short_term['magnitude_probability'] * short_term['RatioPSonTot']).sum()

    if(tmpbs > neg_prob):
        BScomputedYN = True
    if(tmpps > neg_prob):
        PScomputedYN = True

    out['BScomputedYN'] = BScomputedYN
    out['PScomputedYN'] = PScomputedYN

    print(' <-- Negligible Probability: %.4f' % neg_prob)
    print(' --> Sum of probability BS scenarios: %.4f  --> Compute Probability Scenarios: %r' % (tmpbs, BScomputedYN))
    print(' --> Sum of probability PS scenarios: %.4f  --> Compute Probability Scenarios: %r' % (tmpps, PScomputedYN))

    return out

def bs_probability_scenarios(**kwargs):

    Config          = kwargs.get('cfg', None)
    short_term      = kwargs.get('short_term', None)
    prob_scenes     = kwargs.get('prob_scenes', None)
    samp_ens        = kwargs.get('samp_ens', None)
    pre_selection   = kwargs.get('pre_selection', None)
    Discretizations = kwargs.get('Discretizations', None)
    region_files    = kwargs.get('regions_files', None)
    NBS	            = kwargs.get('NBS', None)
    Nid             = kwargs.get('Nid', None)
    int_ens         = kwargs.get('intervals_ensemble', None)
    region_info     = dict()


    if(samp_ens[Nid]['BScomputedYN'] == False or short_term['BS_computed_YN'] == False or pre_selection['BS_scenarios'] == False):

        samp_ens[Nid]['nr_bs_scenarios'] = 0

        return samp_ens

    regions_nr = []

#    iScenBS = 0
#    sel_mag = len(pre_selection['sel_BS_Mag_idx'])
#    bs2_pos = len(pre_selection['BS2_Position_Selection_common'])
#    foc_ids = len(Discretizations['BS-4_FocalMechanism']['ID'])
#
#    import copy
#
#    for i1 in range(sel_mag):
#        imag = pre_selection['sel_BS_Mag_idx'][i1]
#
#        for i2 in range(bs2_pos):
#            ipos = pre_selection['BS2_Position_Selection_common'][i2]
#            ireg = Discretizations['BS-2_Position']['Region'][ipos]
#
#            # Faccio il load della regione se non già fatto
#            if(ireg not in regions_nr):
#
#                #print("...............................", region_files)
#                region_info = load_region_infos(ireg         = ireg,
#                                                region_info  = region_info,
#                                                region_files = region_files)
#                regions_nr.append(ireg)
#
#            #RegMeanProb_BS4 = region_info[ireg]['BS4_FocMech_MeanProb_val']
#            RegMeanProb_BS4 = region_info[ireg]['BS4_FocMech_MeanProb_valNorm']
#            #print(RegMeanProb_BS4)
#            #print(np.shape(RegMeanProb_BS4))
#            #print(ireg)
#            #return
#
#
    ## seed random number generator
    random_value = np.random.random(NBS)
    iscenbs=0
    #ratio=samp_ens[Nid]['nr_bs_scenarios']/short_term['Total_BS_Scenarios']
    for i in random_value:
        idx,proba = find_nearest(int_ens,i)
        samp_ens[Nid]['iscenbs'][iscenbs]=idx
        samp_ens[Nid]['prob_scenarios_bs'][iscenbs]=prob_scenes['ProbScenBS'][idx]
        for j in range(5):
            samp_ens[Nid]['prob_scenarios_bs_fact'][iscenbs,j]=prob_scenes['prob_scenarios_bs_fact'][idx,j]
        for j in range(11):
            samp_ens[Nid]['par_scenarios_bs'][iscenbs,j]=prob_scenes['par_scenarios_bs'][idx,j] 
        iscenbs=iscenbs+1

    samp_ens[Nid]['nr_bs_scenarios'] = np.shape(samp_ens[Nid]['prob_scenarios_bs_fact'])[0]
    #print(NBS)
    #print(len(samp_ens[Nid]['par_scenarios_bs'][0,:]))
    #print(len(samp_ens[Nid]['par_scenarios_bs'][:,0]))
    return samp_ens


def ps_probability_scenarios(**kwargs):

    Config             = kwargs.get('cfg', None)
    short_term         = kwargs.get('short_term', None)
    prob_scenes        = kwargs.get('prob_scenes', None)
    samp_ens           = kwargs.get('samp_ens', None)
    pre_selection      = kwargs.get('pre_selection', None)
    Model_Weights      = kwargs.get('Model_Weights', None)
    regions            = kwargs.get('regions', None)
    PSBarInfo          = kwargs.get('PSBarInfo', None)
    region_ps          = kwargs.get('region_ps', None)
    Scenarios_PS       = kwargs.get('Scenarios_PS', None)
    ps1_magnitude      = kwargs.get('ps1_magnitude', None)
    lambda_bsps        = kwargs.get('lambda_bsps', None)
    Nid                = kwargs.get('Nid', None)

    samp_ens[Nid]['PScomputedYN'] == False
    
    if(samp_ens[Nid]['PScomputedYN'] == False or short_term['PS_computed_YN'] == False):

        #print("--------uuuuuuuuuuuu------------>>", samp_ens[Nid]['PScomputedYN'], short_term['PS_computed_YN'])

        #fbfix 2021-11-26
        samp_ens[Nid]['PScomputedYN']    = False
        short_term['PS_computed_YN']   = False
        #
        samp_ens[Nid]['nr_ps_scenarios'] = 0

        return samp_ens

    iScenPS = 0

    print("Select PS Probability Scenarios")

    sel_mag = len(pre_selection['sel_PS_Mag_idx'][0])
    sel_imod= len(Model_Weights['PS2_Bar']['Wei'])

    par_scenarios_ps           = np.zeros((0,7))
    prob_scenarios_ps_fact     = np.zeros((0,5))
    iScenPS_scenarios_ps       = np.zeros((0))
    nrireg = 0

    #print(Scenarios_PS[3]['SlipDistribution'])
    #return

    aa=0
    bb=0
    for i1 in range(sel_mag):
        imag = pre_selection['sel_PS_Mag_idx'][0][i1]
        Imag = pre_selection['sel_PS_Mag_val'][i1]
        IMAG = imag+1

        tmp_b = np.where(short_term['PS_model_YN'][imag] == 1)
        #print(short_term['PS_model_YN'])
        #print(short_term['PS_model_YN'][imag])
        #print(imag, tmp_b)

        for imod in range(sel_imod-1):

            ps_modedls = pre_selection['Inside_in_BarPSperModel'][imag][imod]['inside']

            if(len(ps_modedls) == 0):
                continue

            for i2 in range(len(ps_modedls)):


                ibar   = -1
                ibar   = pre_selection['Inside_in_BarPSperModel'][imag][imod]['inside'][i2]

                if(imod == -1):
                    print('-----------------------',ibar, imag, imod)
                    print('...',PSBarInfo['BarPSperModelReg'][imag][imod], type(PSBarInfo['BarPSperModelReg'][imag][imod]), ibar)
                    print(PSBarInfo['BarPSperModelReg'][imag][imod])
                    print('...',PSBarInfo['BarPSperModelReg'][imag])

                try:
                    ireg   = PSBarInfo['BarPSperModelReg'][imag][imod][ibar][0]
                except:
                    try:
                        ireg   = PSBarInfo['BarPSperModelReg'][imag][imod][ibar]
                    except:
                        ireg   = np.int64(PSBarInfo['BarPSperModelReg'][imag][imod])

                try:
                    nr_reg = region_ps[ireg-1]
                except:
                    bb=bb+1
                    continue

                tmp_a = int(lambda_bsps['regionsPerPS'][ireg-1])

                if(ireg == 3 or ireg == 44 or ireg == 48):
                    continue


                if(short_term['sel_RatioPSonPSTot'][nr_reg-1] > 0):


                    selected_maPsIndex        = np.where(Scenarios_PS[ireg]['magPSInd'] == imag+1)  #### PORCODDDDDIOOOOOOOOOOOOO Gli indici che iniziano da 1.... non siamo al supermercato!!!!!!!!
                    selected_SlipDistribution = np.take(Scenarios_PS[ireg]['SlipDistribution'],selected_maPsIndex)
                    slipVal                   = np.unique(selected_SlipDistribution)
                    nScen                     = len(slipVal)
                    locScen                   = np.array(range(iScenPS +1, iScenPS + nScen))
                    vectmp                    = np.ones(nScen)


                    if(len(vectmp) == 0):
                        return False

                    tmp_par_scenarios_ps       = np.zeros((len(vectmp), 7))
                    tmp_prob_scenarios_ps_fact = np.zeros((len(vectmp), 5))
                    tmp_iScenPS_scenarios_ps   = np.zeros((len(vectmp), 0)) 

                    for k in range(len(vectmp)):

                        tmp_par_scenarios_ps[k][0]       = vectmp[k] * ireg
                        tmp_par_scenarios_ps[k][1]       = vectmp[k] * ps1_magnitude['Val'][imag]
                        tmp_par_scenarios_ps[k][2]       = vectmp[k] * PSBarInfo['BarPSperModel'][imag][imod]['pos_xx'][ibar]
                        tmp_par_scenarios_ps[k][3]       = vectmp[k] * PSBarInfo['BarPSperModel'][imag][imod]['pos_yy'][ibar]
                        tmp_par_scenarios_ps[k][4]       = vectmp[k] * imod +1 #always matlab start from 1
                        tmp_par_scenarios_ps[k][5]       = slipVal[k]

                        try:
                            tmp_par_scenarios_ps[k][6]       = vectmp[k] * PSBarInfo['BarPSperModelDepth'][imag][imod][ibar]
                        except:
                            tmp_par_scenarios_ps[k][6]       = vectmp[k] * PSBarInfo['BarPSperModelDepth'][imag][imod]

                        tmp_prob_scenarios_ps_fact[k][0] = vectmp[k] * short_term['magnitude_probability'][imag]
                        tmp_prob_scenarios_ps_fact[k][1] = vectmp[k] * short_term['BarProb'][imod][imag][i2]
                        tmp_prob_scenarios_ps_fact[k][2] = vectmp[k] * short_term['RatioPSonTot'][imag] * short_term['RatioPSonTot'][tmp_a]
                        tmp_prob_scenarios_ps_fact[k][3] = vectmp[k] * Model_Weights['PS2_Bar']['Wei'][imod] / np.sum(Model_Weights['PS2_Bar']['Wei'][tmp_b])
                        tmp_prob_scenarios_ps_fact[k][4] = vectmp[k] / nScen
                        
                        tmp_iScenPS_scenarios_ps[k] = vectmp[k] * k

                    prob_scenarios_ps_fact = np.concatenate((prob_scenarios_ps_fact, tmp_prob_scenarios_ps_fact), axis=0)
                    par_scenarios_ps       = np.concatenate((par_scenarios_ps, tmp_par_scenarios_ps), axis=0)
                    iScenPS_scenarios_ps   = np.concatenate((iScenPS_scenarios_ps, tmp_iScenPS_scenarios_ps), axis=0)
                    iScenPS = iScenPS + nScen

                    ####
                    #if ireg == 49:
                    #    aa=aa+1
                    #    nrireg = nrireg+1

    samp_ens[Nid]['iscenps'] = iScenPS_scenarios_ps
    samp_ens[Nid]['prob_scenarios_ps_fact'] = prob_scenarios_ps_fact
    samp_ens[Nid]['par_scenarios_ps']       = par_scenarios_ps
    samp_ens[Nid]['nr_ps_scenarios']        = np.shape(samp_ens[Nid]['prob_scenarios_ps_fact'])[0]

    print(np.shape(par_scenarios_ps), np.shape(prob_scenarios_ps_fact),bb)

    return prob_scenes


def load_region_infos(**kwargs):

    ireg        = kwargs.get('ireg', None)
    files       = kwargs.get('region_files', None)
    region_info = kwargs.get('region_info', None)

    info = np.load(files['ModelsProb_Region_files'][ireg-1], allow_pickle=True).item()

    region_info[ireg] = info

    return region_info

