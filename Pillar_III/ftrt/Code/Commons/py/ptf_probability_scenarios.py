
import numpy as np

def compute_probability_scenarios(**kwargs):


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


    # Check if Xs schenarios prob must be computed and inizialize dictionary
    probability_scenarios = set_if_compute_scenarios(cfg           = Config,
                                                     short_term    = short_term,
                                                     pre_selection = pre_selection)

    #Some inizializations
    probability_scenarios['par_scenarios_bs']       = np.zeros( (int(short_term['Total_BS_Scenarios']), 11) )
    probability_scenarios['prob_scenarios_bs_fact'] = np.zeros( (int(short_term['Total_BS_Scenarios']),  5) )
    probability_scenarios['par_scenarios_ps']       = np.zeros( (int(short_term['Total_PS_Scenarios']),  7) )
    probability_scenarios['prob_scenarios_ps_fact'] = np.zeros( (int(short_term['Total_PS_Scenarios']),  5) )


    probability_scenarios = bs_probability_scenarios(cfg              = Config,
                                                     short_term       = short_term,
                                                     pre_selection    = pre_selection,
                                                     regions_files    = regions,
                                                     prob_scenes      = probability_scenarios,
                                                     Discretizations  = LongTermInfo['Discretizations'])

    probability_scenarios = ps_probability_scenarios(cfg              = Config,
                                                     PSBarInfo        = PSBarInfo,
                                                     short_term       = short_term,
                                                     pre_selection    = pre_selection,
                                                     prob_scenes      = probability_scenarios,
                                                     region_ps        = LongTermInfo['region_listPs'],
                                                     Model_Weights    = LongTermInfo['Model_Weights'],
                                                     Scenarios_PS     = Scenarios_PS,
                                                     ps1_magnitude    = LongTermInfo['Discretizations']['PS-1_Magnitude'],
                                                     lambda_bsps      = lambda_bsps)

    if(probability_scenarios == False):
        return False

    # check on the nr scenarios computed into the two section. Should be identical
    check_bs = 'OK'
    check_ps = 'OK'

    if(probability_scenarios['nr_bs_scenarios'] != short_term['Total_BS_Scenarios']):
        check_bs = 'WARNING'
    if(probability_scenarios['nr_ps_scenarios'] != short_term['Total_PS_Scenarios']):
        check_ps = 'WARNING'

    print(' --> Check Nr Bs scenarios: %7d  <--> %7d --> %s' % (probability_scenarios['nr_bs_scenarios'], short_term['Total_BS_Scenarios'], check_bs))
    print(' --> Check Nr Ps scenarios: %7d  <--> %7d --> %s' % (probability_scenarios['nr_ps_scenarios'], short_term['Total_PS_Scenarios'], check_ps))

    probability_scenarios['relevant_scenarios_bs'] = np.unique(probability_scenarios['par_scenarios_bs'][:,0])
    probability_scenarios['relevant_scenarios_ps'] = np.unique(probability_scenarios['par_scenarios_ps'][:,0])
    #print(probability_scenarios['relevant_scenarios_ps'],'\n\n')
    #return

    print(' --> Relevant Scenarios BS : ', probability_scenarios['relevant_scenarios_bs'])
    print(' --> Relevant Scenarios PS : ', probability_scenarios['relevant_scenarios_ps'])

    # Re-Normalize scenarios (to manage events outside nSigma, BS for large events, ...)
    ProbScenBS        = probability_scenarios['prob_scenarios_bs_fact'].prod(axis=1)
    print(np.sum(ProbScenBS))
    ProbScenPS        = probability_scenarios['prob_scenarios_ps_fact'].prod(axis=1)
    TotProbBS_preNorm = np.sum(ProbScenBS)
    TotProbPS_preNorm = np.sum(ProbScenPS)
    TotProb_preNorm   = TotProbBS_preNorm + TotProbPS_preNorm

    # No scenarios bs or ps possible
    if(TotProb_preNorm == 0):
        return False

    if(TotProb_preNorm < 1.0):
        ProbScenBS = ProbScenBS / TotProb_preNorm
        ProbScenPS = ProbScenPS / TotProb_preNorm
        print(' --> Total Bs scenarios probability pre-renormalization: %.5f' % TotProbBS_preNorm)
        print(' --> Total Ps scenarios probability pre-renormalization: %.5f' % TotProbPS_preNorm)
        print('     --> Total Bs and Ps probabilty renormalized to 1')

    probability_scenarios['ProbScenBS'] = ProbScenBS
    probability_scenarios['ProbScenPS'] = ProbScenPS

    print('############ TOT BS ##############',TotProbBS_preNorm)
    print('############ TOT prb ##############',TotProb_preNorm)
    print('############ TOT sum ##############',np.sum(ProbScenBS))


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

    probability_scenarios['best_scenarios'] = {'max_idxBS':max_idxBS, 'max_idxPS':max_idxPS, 'max_ValBS':max_ValBS, 'max_ValPS':max_ValPS}

    print('     --> Best Bs scenario Idx and Value: %6d    %.5e' % (max_idxBS, max_ValBS))
    print('     --> Best Ps scenario Idx and Value: %6d    %.5e' % (max_idxPS, max_ValPS))

    return probability_scenarios

def bs_probability_scenarios(**kwargs):

    Config          = kwargs.get('cfg', None)
    short_term      = kwargs.get('short_term', None)
    prob_scenes     = kwargs.get('prob_scenes', None)
    pre_selection   = kwargs.get('pre_selection', None)
    Discretizations = kwargs.get('Discretizations', None)
    region_files    = kwargs.get('regions_files', None)

    region_info     = dict()


    if(prob_scenes['BScomputedYN'] == False or short_term['BS_computed_YN'] == False or pre_selection['BS_scenarios'] == False):

        prob_scenes['nr_bs_scenarios'] = 0

        return prob_scenes

    print("Select BS Probability Scenarios")


    regions_nr = []

    iScenBS = 0
    sel_mag = len(pre_selection['sel_BS_Mag_idx'])
    bs2_pos = len(pre_selection['BS2_Position_Selection_common'])
    foc_ids = len(Discretizations['BS-4_FocalMechanism']['ID'])

    import copy

    for i1 in range(sel_mag):
        imag = pre_selection['sel_BS_Mag_idx'][i1]

        for i2 in range(bs2_pos):
            ipos = pre_selection['BS2_Position_Selection_common'][i2]
            ireg = Discretizations['BS-2_Position']['Region'][ipos]

            # Faccio il load della regione se non già fatto
            if(ireg not in regions_nr):

                #print("...............................", region_files)
                region_info = load_region_infos(ireg         = ireg,
                                                region_info  = region_info,
                                                region_files = region_files)
                regions_nr.append(ireg)

            #RegMeanProb_BS4 = region_info[ireg]['BS4_FocMech_MeanProb_val']
            RegMeanProb_BS4 = region_info[ireg]['BS4_FocMech_MeanProb_valNorm']
            #print(RegMeanProb_BS4)
            #print(np.shape(RegMeanProb_BS4))
            #print(ireg)
            #return


            # Non credo che qui ci saranno errori, nel senso che i npy sono creati a partire dai mat conteneti roba
            if(RegMeanProb_BS4.size == 0):
                 print(' --> WARNING: region info %d is empty!!!' % (ireg) )


            ipos_reg = np.where(region_info[ireg]['BS4_FocMech_iPosInRegion'] == ipos+1)[1]
            tmpProbAngles = RegMeanProb_BS4[ipos_reg[0]]

            len_depth_valvec = len(Discretizations['BS-3_Depth']['ValVec'][imag][ipos])

            #I3 (depth) AND I4 (angles)  ENUMERATE ALL RELEVANT SCENARIOS FOR EACH MAG AND POS (Equivalent to compute_scenarios_prefixes)
            for i3 in range(len_depth_valvec):

                for i4 in range(foc_ids):

                    lon, lat          = Discretizations['BS-2_Position']['Val'][ipos].split()
                    strike, dip, rake = Discretizations['BS-4_FocalMechanism']['Val'][i4].split()

                    prob_scenes['par_scenarios_bs'][iScenBS][0]       = int(ireg)
                    prob_scenes['par_scenarios_bs'][iScenBS][1]       = Discretizations['BS-1_Magnitude']['Val'][imag]
                    prob_scenes['par_scenarios_bs'][iScenBS][2]       = float(lon)
                    prob_scenes['par_scenarios_bs'][iScenBS][3]       = float(lat)
                    prob_scenes['par_scenarios_bs'][iScenBS][4]       = Discretizations['BS-3_Depth']['ValVec'][imag][ipos][i3]
                    prob_scenes['par_scenarios_bs'][iScenBS][5]       = float(strike)
                    prob_scenes['par_scenarios_bs'][iScenBS][6]       = float(dip)
                    prob_scenes['par_scenarios_bs'][iScenBS][7]       = float(rake)
                    prob_scenes['par_scenarios_bs'][iScenBS][8]       = Discretizations['BS-5_Area']['ValArea'][ireg, imag, i4]
                    prob_scenes['par_scenarios_bs'][iScenBS][9]       = Discretizations['BS-5_Area']['ValLen'][ireg, imag, i4]
                    prob_scenes['par_scenarios_bs'][iScenBS][10]      = Discretizations['BS-6_Slip']['Val'][ireg, imag, i4]

                    prob_scenes['prob_scenarios_bs_fact'][iScenBS][0] = short_term['magnitude_probability'][imag]
                    prob_scenes['prob_scenarios_bs_fact'][iScenBS][1] = short_term['PosProb'][i1, i2]
                    prob_scenes['prob_scenarios_bs_fact'][iScenBS][2] = short_term['RatioBSonTot'][imag]
                    prob_scenes['prob_scenarios_bs_fact'][iScenBS][3] = short_term['DepProbScenes'][i1, i2][i3]
                    prob_scenes['prob_scenarios_bs_fact'][iScenBS][4] = tmpProbAngles[i4]

                    iScenBS = iScenBS + 1

    prob_scenes['nr_bs_scenarios'] = np.shape(prob_scenes['prob_scenarios_bs_fact'])[0]


    return prob_scenes


def load_region_infos(**kwargs):

    ireg        = kwargs.get('ireg', None)
    files       = kwargs.get('region_files', None)
    region_info = kwargs.get('region_info', None)

    info = np.load(files['ModelsProb_Region_files'][ireg-1], allow_pickle=True).item()

    region_info[ireg] = info

    return region_info

def ps_probability_scenarios(**kwargs):


    Config             = kwargs.get('cfg', None)
    short_term         = kwargs.get('short_term', None)
    prob_scenes        = kwargs.get('prob_scenes', None)
    pre_selection      = kwargs.get('pre_selection', None)
    Model_Weights      = kwargs.get('Model_Weights', None)
    regions            = kwargs.get('regions', None)
    PSBarInfo          = kwargs.get('PSBarInfo', None)
    region_ps          = kwargs.get('region_ps', None)
    Scenarios_PS       = kwargs.get('Scenarios_PS', None)
    ps1_magnitude      = kwargs.get('ps1_magnitude', None)
    lambda_bsps        = kwargs.get('lambda_bsps', None)

    if(prob_scenes['PScomputedYN'] == False or short_term['PS_computed_YN'] == False):

        #print("--------uuuuuuuuuuuu------------>>", prob_scenes['PScomputedYN'], short_term['PS_computed_YN'])

        #fbfix 2021-11-26
        prob_scenes['PScomputedYN']    = False
        short_term['PS_computed_YN']   = False
        #
        prob_scenes['nr_ps_scenarios'] = 0

        return prob_scenes

    iScenPS = 0

    print("Select PS Probability Scenarios")


    sel_mag = len(pre_selection['sel_PS_Mag_idx'][0])
    sel_imod= len(Model_Weights['PS2_Bar']['Wei'])

    par_scenarios_ps           = np.zeros((0,7))
    prob_scenarios_ps_fact     = np.zeros((0,5))
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

                    for k in range(len(vectmp)):

                        # fb 2022-04-21
                        # To fix approximation errors: use round(x,5) for 2 and 3 elements
                        # THIS may be very weak. Better to think new algorithm
                        tmp_par_scenarios_ps[k][0]       = vectmp[k] * ireg
                        tmp_par_scenarios_ps[k][1]       = vectmp[k] * ps1_magnitude['Val'][imag]
                        tmp_par_scenarios_ps[k][2]       = round(vectmp[k] * PSBarInfo['BarPSperModel'][imag][imod]['pos_xx'][ibar],5)
                        tmp_par_scenarios_ps[k][3]       = round(vectmp[k] * PSBarInfo['BarPSperModel'][imag][imod]['pos_yy'][ibar],5)
                        tmp_par_scenarios_ps[k][4]       = round(vectmp[k] * imod +1,5) #always matlab start from 1
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

                    prob_scenarios_ps_fact = np.concatenate((prob_scenarios_ps_fact, tmp_prob_scenarios_ps_fact), axis=0)
                    par_scenarios_ps       = np.concatenate((par_scenarios_ps, tmp_par_scenarios_ps), axis=0)

                    iScenPS = iScenPS + nScen

                    ####
                    #if ireg == 49:
                    #    aa=aa+1
                    #    nrireg = nrireg+1
                    #    #print(">>>> ", aa, imag, Imag, imod, i2, ireg, nrireg, '--', np.shape(par_scenarios_ps), ' *** ', np.shape(prob_scenarios_ps_fact), len(vectmp), slipVal)
                    #    print(">>>> ",ireg, imag, aa, locScen, np.shape(vectmp), Imag, np.shape(par_scenarios_ps), np.shape(prob_scenarios_ps_fact))


    prob_scenes['prob_scenarios_ps_fact'] = prob_scenarios_ps_fact
    prob_scenes['par_scenarios_ps']       = par_scenarios_ps
    prob_scenes['nr_ps_scenarios']        = np.shape(prob_scenes['prob_scenarios_ps_fact'])[0]

    print(np.shape(par_scenarios_ps), np.shape(prob_scenarios_ps_fact),bb)

    return prob_scenes


def set_if_compute_scenarios(**kwargs):

    short_term     = kwargs.get('short_term', None)
    pre_selection  = kwargs.get('pre_selection', None)
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
