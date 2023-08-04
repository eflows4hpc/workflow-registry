import numpy as np
from scipy.stats       import norm
from random import random
import utm
import random
from ismember          import ismember
from ptf_scaling_laws  import correct_BS_vertical_position
from ptf_scaling_laws  import correct_BS_horizontal_position
import scipy
import copy
import math
from math import radians, cos, sin, asin, sqrt

def compute_ensemble_sampling_RS(**kwargs):

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

    discretizations=LongTermInfo['Discretizations']
    TotProbBS_all = 1.0 #np.sum(probability_scenarios['ProbScenBS'])
    TotProbPS_all = 0.0 #np.sum(probability_scenarios['ProbScenPS'])
   
    RS_type = Config.get('Sampling','RS_type')
    short_term = TotProbBS_all
    ### Number of scenarios and runs ###
    RS_samp_scen=int(Config.get('Sampling','RS_samp_scen'))
    RS_samp_run=int(Config.get('Sampling','RS_samp_run'))
    Nsamp=np.ones(RS_samp_run)*RS_samp_scen
    sampled_ensemble = {}

    Nid=0
    ### Loop of Nsamp MC sampling ###
    for N in Nsamp:
        ### Begining of the creation of the Nth sampled ensemble ###
        N=int(N)
        NBS= round(TotProbBS_all*N) ### NBS: number of scenarios sampled from the BS ensemble
        NPS= N-NBS ### NPS: number of scenarios sampled from the PS ensemble
        # NBS and NPS are proportionnal to the probability of PS and BS
        sampled_ensemble[Nid] = dict()
        sampled_ensemble[Nid] = set_if_compute_scenarios(cfg           = Config,
                                                short_term    = short_term)
        
       
        ### Initialization of the dictionnaries ### 
        sampled_ensemble[Nid]['prob_scenarios_bs_fact'] = np.zeros( (NBS,  5) )
        sampled_ensemble[Nid]['prob_scenarios_bs'] = np.zeros( (NBS) )
        sampled_ensemble[Nid]['prob_angles_bs'] = np.zeros( (NBS) )
        sampled_ensemble[Nid]['real_prob_scenarios_bs'] = np.zeros( (NBS) )
        sampled_ensemble[Nid]['par_scenarios_bs'] = np.zeros(  (NBS, 11) )
        sampled_ensemble[Nid]['real_par_scenarios_bs'] = np.zeros(  (NBS, 11) )
        sampled_ensemble[Nid]['prob_scenarios_ps_fact'] = np.zeros( (NPS,  5) )
        sampled_ensemble[Nid]['prob_scenarios_ps'] = np.zeros( (NPS) )
        sampled_ensemble[Nid]['real_prob_scenarios_ps'] = np.zeros( (NPS) )
        sampled_ensemble[Nid]['par_scenarios_ps'] = np.zeros(  (NPS,  7) )
        sampled_ensemble[Nid]['real_par_scenarios_ps'] = np.zeros( (NPS, 7) )
        sampled_ensemble[Nid]['iscenbs']=np.zeros(NBS)
        sampled_ensemble[Nid]['iscenps']=np.zeros(NPS)

        if RS_type == 'MC':
           sampled_ensemble = bs_probability_scenarios_RMCS(cfg              = Config,
                                                            short_term       = short_term,
                                                            pre_selection    = pre_selection,
                                                            regions_files    = regions,
                                                            prob_scenes      = probability_scenarios,
                                                            samp_ens         = sampled_ensemble,
                                                            Discretizations  = LongTermInfo['Discretizations'],
	   					            NBS	      = NBS,
                                                            Nid	      = Nid,
                                                            Num_Samp         = NBS,
                                                            ee               = ee)

        if RS_type == 'LH':
           sampled_ensemble = bs_probability_scenarios_RLHS(cfg              = Config,
                                                            short_term       = short_term,
                                                            pre_selection    = pre_selection,
                                                            regions_files    = regions,
                                                            prob_scenes      = probability_scenarios,
                                                            samp_ens         = sampled_ensemble,
                                                            Discretizations  = LongTermInfo['Discretizations'],
                                                            NBS              = NBS,
                                                            Nid              = Nid,
                                                            Num_Samp         = NBS,
                                                            ee               = ee)



        if(sampled_ensemble[Nid] == False):
            return False

        TotBS=len(sampled_ensemble[Nid]['real_par_scenarios_bs'])
        TotPS=len(sampled_ensemble[Nid]['real_par_scenarios_ps'])
        Tot=TotBS+TotPS
        sampled_ensemble[Nid]['RealProbScenBS'] = np.ones(TotBS)*1./Tot
        sampled_ensemble[Nid]['RealProbScenPS'] = np.ones(TotPS)*1./Tot
        
        ### Bayes approach ###
        #ang_prob = (np.ones(TotBS)*1./Tot)/sampled_ensemble[Nid]['prob_angles_bs']
        #sampled_ensemble[Nid]['RealProbScenBS'] = ang_prob/np.sum(ang_prob)
        
        ### Re-normalization of all the probabilities ###
        TotProbBS = np.sum(sampled_ensemble[Nid]['RealProbScenBS'])
        TotProbPS = np.sum(sampled_ensemble[Nid]['RealProbScenPS'])
        sampled_ensemble[Nid]['real_relevant_scenarios_bs'] = np.unique(sampled_ensemble[Nid]['real_par_scenarios_bs'][:,0])
        print("Final probabilities",TotProbBS,TotProbPS)

        try:
            max_idxBS = np.argmax(RealProbScenBS)
        except:
            max_idxBS = -1
        try:
            max_ValBS = ProbScenBS[max_idxBS]
        except:
            max_ValBS = 0
        try:
            max_idxPS = np.argmax(RealProbScenPS)
        except:
            max_idxPS = -1
        try:
            max_ValPS = ProbScenPS[max_idxPS]
        except:
            max_ValPS = 0

            max_ValPS = 0

            max_ValPS = 0

        sampled_ensemble[Nid]['real_best_scenarios'] = {'max_idxBS':max_idxBS, 'max_idxPS':max_idxPS, 'max_ValBS':max_ValBS, 'max_ValPS':max_ValPS}

        Nid=Nid+1

    return sampled_ensemble

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
    
    tmpbs      = short_term #(short_term['magnitude_probability'] * short_term['RatioBSonTot']).sum()
    tmpps      = 1.0-short_term #(short_term['magnitude_probability'] * short_term['RatioPSonTot']).sum()

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

## Function giving random values of rake ##
def roll(massDist):
        randRoll = random.random() # in [0,1]
        sum = 0
        result = 0
        for mass in massDist:
            sum += mass
            if randRoll < sum:
                return result
            result+=90

def find_nearest(array, value):
    arr = np.asarray(array)
    idx = 0
    diff = arr-value
    diff[diff<1e-26]=100.0
    idx=diff.argmin()
    return idx,array[idx]

def bs_probability_scenarios_RMCS(**kwargs):

    Config          = kwargs.get('cfg', None)
    short_term      = kwargs.get('short_term', None)
    prob_scenes     = kwargs.get('prob_scenes', None)
    samp_ens        = kwargs.get('samp_ens', None)
    pre_selection   = kwargs.get('pre_selection', None)
    Discretizations = kwargs.get('Discretizations', None)
    region_files    = kwargs.get('regions_files', None)
    NBS	            = kwargs.get('NBS', None)
    Nid             = kwargs.get('Nid', None)
    Num_Samp        = kwargs.get('Num_Samp', None)
    ee              = kwargs.get('ee', None)
    region_info     = dict()
    ee_d=ee

    
    #if(samp_ens[Nid]['BScomputedYN'] == False or short_term['BS_computed_YN'] == False or pre_selection['BS_scenarios'] == False):
    if(samp_ens[Nid]['BScomputedYN'] == False or pre_selection['BS_scenarios'] == False):
        samp_ens[Nid]['nr_bs_scenarios'] = 0
        return samp_ens

    regions_nr = []

    # Parameters and arrays
    Mw=ee['mag']
    sig=ee['MagSigma']
    mu      = ee['PosMean_3d']
    #co      = copy.deepcopy(ee['PosCovMat_3dm'])
    hx             = ee['ee_utm'][0]
    hy             = ee['ee_utm'][1]
    hz             = ee['depth']* (1000.0)
    xyz            = np.array([hx, hy, hz])
    real_val=np.zeros((Num_Samp,11))
    disc_val=np.zeros((Num_Samp,11))
    iscenbs=0

    #myfile = open('xyz.txt', 'w')
    
    ### Beginning of the parameter selection for each scenario ###
    for i in range(Num_Samp):
       
       print(i)
       iscenbs=i

       ### Choice of the magnitude ###
       ###############################
       mag_val_disc = 0.0
       mag_val=np.random.normal(loc=Mw, scale=sig)
       mag_min_presel=np.ndarray.min(pre_selection['sel_BS_Mag_val'][:])
       if mag_val<mag_min_presel:
           mag_val_disc=mag_min_presel
       else:
           mag_val_disc=mag_val
       
       ### Compute vertical half_width with respect the magnitude ###
       v_hwidth = correct_BS_vertical_position(mag = mag_val)
       h_hwidth = correct_BS_horizontal_position(mag = mag_val)
       area = (2.0*v_hwidth/(math.sin(math.pi/4)))*(2.0*h_hwidth)
       length = 2.0*h_hwidth
       rig = 30.0e9
       ### Moment ###
       Mo=10**((mag_val+10.7)*(3.0/2.0))*1e-7
       slip=Mo/(area*rig)
       
       ### Choice of the position (lat, lon, depth) ###
       ################################################

       # Correct  Covariance matrix ###
       PosCovMat_3d    = np.array([[ee_d['cov_matrix']['XX'], ee_d['cov_matrix']['XY'], ee_d['cov_matrix']['XZ']], \
                                     [ee_d['cov_matrix']['XY'], ee_d['cov_matrix']['YY'], ee_d['cov_matrix']['YZ']], \
                                     [ee_d['cov_matrix']['XZ'], ee_d['cov_matrix']['YZ'], ee_d['cov_matrix']['ZZ']]])
       co = PosCovMat_3d*1000000
       cov = copy.deepcopy(co) #np.zeros((3,3))
       cov[0,0] = co[0,0] + h_hwidth**2
       cov[1,1] = co[1,1] + h_hwidth**2
       cov[2,2] = co[2,2] + v_hwidth**2
       mean=xyz
      
       #Definition of the resolution of the gaussian
       bounds=np.zeros((3,2))
       ee_d=copy.deepcopy(ee) 
       #distlim=10000.0*1000.0
       dip_min=math.radians(float(10.0))
       deptlim=50.0*1000.0
       bounds[0,0]=xyz[0]
       bounds[0,1]=xyz[0]
       bounds[1,0]=xyz[1]
       bounds[1,1]=xyz[1]
       bounds[2,0]=1000.0+(v_hwidth*math.sin(dip_min))/math.sin(math.pi/4)
       bounds[2,1]=deptlim 
       Nsteps=1000
       x = np.random.multivariate_normal(mean=mean, cov=cov, size=Nsteps)
       #print('X length check',len(x),mean,cov)
       prob = np.zeros((len(x)))
       prob = scipy.stats.multivariate_normal.pdf(x,mean=xyz, cov=cov)
       izero=[]
       for iii in range(len(x)):
           if np.any(x[iii,2] < bounds[2,0]) or np.any(x[iii,2] > bounds[2,1]):
              #print('No empty izero') 
              izero=np.append(izero,int(iii))
       if len(izero)>0:
          izero=izero.astype(int)
          prob=np.delete(prob,izero)
          x=np.delete(x,izero,axis=0)
       int_ens = np.zeros(len(x))
       prob_mod = np.zeros(len(x))
       prob_cum = 0
       prob_mod=prob[:]/np.sum(prob)
       for iii in range(len(prob)):
           prob_cum=prob_cum+prob_mod[iii]
           int_ens[iii]= prob_cum
           # Random selection of a value inside the cumulative distrib.
       random_value = np.random.random()
       idx,proba = find_nearest(int_ens,random_value)
       pos_fin=x[idx]
       prob_fin=prob_mod[idx]
       pos_fin=x[0]
       lat_val=pos_fin[0]
       lon_val=pos_fin[1]
       dep_val=pos_fin[2]
       
       # Conversion into Lat/Lon degree values
       if (lat_val>2000000) or (lat_val<0):
           lat_val = 100001
           lon_val = 100001
           latlon = utm.to_latlon(lat_val,
                                  lon_val,
                                  ee['ee_utm'][2], northern=True)
       elif (lat_val<1000000) and (lat_val>100000):
           latlon = utm.to_latlon(lat_val,
                                  lon_val,
                                  ee['ee_utm'][2], northern=True)#ee['ee_utm'][3])
       elif (lat_val>1000000):
           lat_val=100000+(lat_val-999999)
           latlon = utm.to_latlon(lat_val,
                                  lon_val,
                                  ee['ee_utm'][2]+1, northern=True)#ee['ee_utm'][3])
       else:
           lat_val=999999-(100000-lat_val)
           latlon = utm.to_latlon(lat_val,
                                  lon_val,
                                  ee['ee_utm'][2]-1, northern=True)#ee['ee_utm'][3])

       ### Choice of the angles ###
       ############################

       # Inside the original code the strike/dip/rake
       # do not depend of the magnitude and position
       bs2_pos = len(pre_selection['BS2_Position_Selection_inn'])
       d_latlon=np.zeros((bs2_pos,2))
       d_diff=np.zeros((bs2_pos))
       for val in range(bs2_pos):
               tmp_idx = pre_selection['BS2_Position_Selection_inn'][val]
               d_latlon[val,:] = Discretizations['BS-2_Position']['Val'][tmp_idx].split()
               d_diff[val] = haversine(latlon[1], latlon[0], d_latlon[val,0], d_latlon[val,1])
       ipos_idx = int(np.argmin(d_diff))
       ipos = pre_selection['BS2_Position_Selection_inn'][ipos_idx]
       ireg = Discretizations['BS-2_Position']['Region'][ipos]       

       # Faccio il load della regione se non già fatto
       if(ireg not in regions_nr):
           #print("...............................", region_files)
           region_info = load_region_infos(ireg         = ireg,
                                           region_info  = region_info,
                                           region_files = region_files)
           regions_nr.append(ireg)
       RegMeanProb_BS4 = region_info[ireg]['BS4_FocMech_MeanProb_valNorm']

       # Non credo che qui ci saranno errori, nel senso che i npy sono creati a partire dai mat conteneti roba
       if(RegMeanProb_BS4.size == 0):
            print(' --> WARNING: region info %d is empty!!!' % (ireg) )
       ipos_reg = np.where(region_info[ireg]['BS4_FocMech_iPosInRegion'] == ipos+1)[1]
       tmpProbAngles = RegMeanProb_BS4[ipos_reg[0]]

       # Creation of the array of cumulated probability intervals
       int_ens = np.zeros(len(tmpProbAngles))
       prob_cum = 0
       prob_mod=tmpProbAngles/np.sum(tmpProbAngles)
       for iii in range(len(tmpProbAngles)):
           prob_cum=prob_cum+prob_mod[iii]
           int_ens[iii]= prob_cum
       # Random selection of a value inside the cumulative distrib.
       random_value = np.random.random()
       iangle,proba_a = find_nearest(int_ens,random_value)
       proba_angles=tmpProbAngles[iangle]
       #samp_ens[Nid]['prob_angles_bs'] = proba
       str_val,dip_val,rak_val = Discretizations['BS-4_FocalMechanism']['Val'][iangle].split()
       dip_val_rad=math.radians(float(dip_val))

       ### Choice of the angle for the real sampling ###
       #################################################
       # Strike angle: uniform law #
       low_str=max(0,float(str_val)-22.5)
       high_str=min(360,float(str_val)+22.5)
       str_real_val=np.random.uniform(low=low_str, high=high_str)
       # Rake angle: uniform law #
       low_rak=max(0,float(rak_val)-45.0)
       high_rak=min(360,float(rak_val)+45.0)
       rak_real_val=np.random.uniform(low=low_rak, high=high_rak)
       # Dip angle: uniform law #
       low_dip=max(0,float(dip_val)-45.0)
       high_dip=min(90,float(dip_val)+45.0)
       dip_real_val=np.random.uniform(low=low_dip, high=high_dip)
       dip_real_val_rad=math.radians(float(dip_real_val))

       ### Re-evaluation of the depth: the fault plane should not reach the surface ###
       max_dip=max(float(dip_val),dip_real_val)
       max_dip_rad=math.radians(float(max_dip))
       lim_dep=1000.0+(v_hwidth*math.sin(max_dip_rad))/math.sin(math.pi/4)
       if dep_val<lim_dep:
          dep_val=lim_dep

       ### Slip, mag and region ###
       ############################

       mag_list=np.array(Discretizations['BS-1_Magnitude']['Val'][:])
       imag = np.argmin(np.abs(mag_list-mag_val_disc)) 
       # Depth of the event #
       idep = np.argmin(np.abs(Discretizations['BS-3_Depth']['ValVec'][imag][ipos]-(dep_val-(v_hwidth*math.sin(dip_val_rad)/math.sin(math.pi/4)))/1000.0))#/math.sin(math.pi/4)))/1000.0))
       are_val = Discretizations['BS-5_Area']['ValArea'][ireg, imag, iangle]
       len_val = Discretizations['BS-5_Area']['ValLen'][ireg, imag, iangle]
       sli_val = Discretizations['BS-6_Slip']['Val'][ireg, imag, iangle]
      
       ### Real sampled parameters ###
       ###############################
       real_val[i,0]=ireg
       real_val[i,1]=mag_val
       real_val[i,2]=latlon[1]
       real_val[i,3]=latlon[0]
       #real_val[i,4]=dep_val/1000.0 #(dep_val-(v_hwidth*math.sin(dip_val_rad)/math.sin(math.pi/4)))/1000.0 #math.sin(math.pi/4)))/1000.0
       real_val[i,4]=(dep_val-(v_hwidth*math.sin(dip_val_rad)/math.sin(math.pi/4)))/1000.0 #math.sin(math.pi/4)))/1000.0
       real_val[i,5]=str_real_val
       real_val[i,6]=dip_real_val
       real_val[i,7]=rak_real_val
       real_val[i,8]=area/(1000.0**2)
       real_val[i,9]=length/1000.0
       real_val[i,10]=slip
       ##############################

       #### Identification of the corresponding discretized parameters ###
       ###################################################################
       temp_val=np.zeros((10))
       temp_val[0]=ireg
       temp_val[1]=Discretizations['BS-1_Magnitude']['Val'][imag] #pre_selection['sel_BS_Mag_val'][imag] #Discretizations['BS-1_Magnitude']['Val'][imag]
       temp_val[2]=d_latlon[ipos_idx,0]
       temp_val[3]=d_latlon[ipos_idx,1]
       temp_val[4]=Discretizations['BS-3_Depth']['ValVec'][imag][ipos][idep]
       temp_val[5]=str_val
       temp_val[6]=dip_val
       temp_val[7]=rak_val
 
       ### Real values ###
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,0]=real_val[i,0] #Reg
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,1]=real_val[i,1] #Mag
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,2]=real_val[i,2] #Lon
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,3]=real_val[i,3] #Lat
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,4]=real_val[i,4] #Depth
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,5]=real_val[i,5] #Strike
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,6]=real_val[i,6] #Dip
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,7]=real_val[i,7] #Rake
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,8]=real_val[i,8] #Area 
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,9]=real_val[i,9] #Length
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,10]=real_val[i,10] #Slip

    samp_ens[Nid]['nr_bs_scenarios'] = np.shape(samp_ens[Nid]['prob_scenarios_bs_fact'])[0]
    return samp_ens

def bs_probability_scenarios_RLHS(**kwargs):

    Config          = kwargs.get('cfg', None)
    short_term      = kwargs.get('short_term', None)
    prob_scenes     = kwargs.get('prob_scenes', None)
    samp_ens        = kwargs.get('samp_ens', None)
    pre_selection   = kwargs.get('pre_selection', None)
    Discretizations = kwargs.get('Discretizations', None)
    region_files    = kwargs.get('regions_files', None)
    NBS	            = kwargs.get('NBS', None)
    Nid             = kwargs.get('Nid', None)
    Num_Samp        = kwargs.get('Num_Samp', None)
    ee              = kwargs.get('ee', None)
    region_info     = dict()
    ee_d=ee

    #if(samp_ens[Nid]['BScomputedYN'] == False or short_term['BS_computed_YN'] == False or pre_selection['BS_scenarios'] == False):
    if(samp_ens[Nid]['BScomputedYN'] == False or pre_selection['BS_scenarios'] == False):
        samp_ens[Nid]['nr_bs_scenarios'] = 0
        return samp_ens

    regions_nr = []

    # Parameters and arrays
    Mw=ee['mag']
    sig=ee['MagSigma']
    mu      = ee['PosMean_3d']
    #co      = copy.deepcopy(ee['PosCovMat_3dm'])
    hx             = ee['ee_utm'][0]
    hy             = ee['ee_utm'][1]
    hz             = ee['depth']* (1000.0)
    xyz            = np.array([hx, hy, hz])
    real_val=np.zeros((Num_Samp,11))
    disc_val=np.zeros((Num_Samp,11))
    iscenbs=0

    ### Calculation of the combined grid of parameters and its associated PDF ###    

    ### Magnitude PDF ###
    #####################
    mag_val_disc = 0.0
    magmax = Mw + 2.0*sig
    magmin = Mw - 2.0*sig
    mag_val_grid = np.arange(magmin,magmax,0.02)
    mag_prob_pdf = scipy.stats.norm.pdf(mag_val_grid,loc=Mw, scale=sig)
    mag_prob_cum = scipy.stats.norm.cdf(mag_val_grid,loc=Mw, scale=sig)
    mag_prob_pdf_norm = mag_prob_pdf/np.sum(mag_prob_pdf)
    ### Correct  Covariance matrix ###
    v_hwidth = correct_BS_vertical_position(mag = Mw)
    h_hwidth = correct_BS_horizontal_position(mag = Mw)

    PosCovMat_3d    = np.array([[ee_d['cov_matrix']['XX'], ee_d['cov_matrix']['XY'], ee_d['cov_matrix']['XZ']], \
                                  [ee_d['cov_matrix']['XY'], ee_d['cov_matrix']['YY'], ee_d['cov_matrix']['YZ']], \
                                  [ee_d['cov_matrix']['XZ'], ee_d['cov_matrix']['YZ'], ee_d['cov_matrix']['ZZ']]])
    co = PosCovMat_3d*1000000
    cov = copy.deepcopy(co) #np.zeros((3,3))
    cov[0,0] = co[0,0] + h_hwidth**2
    cov[1,1] = co[1,1] + h_hwidth**2
    cov[2,2] = co[2,2] + v_hwidth**2
    mean=xyz
    
    ### Position PDF ###
    ####################
    xarrmin = xyz[0] - (2.)*math.sqrt(cov[0,0]) #10000 #co[0,0]
    xarrmax = xyz[0] + (2.)*math.sqrt(cov[0,0]) #10000 #co[0,0]
    xarr = np.arange(xarrmin,xarrmax,5000)
    yarrmin = xyz[1] - (2.)*math.sqrt(cov[1,1]) #10000 #co[1,1]
    yarrmax = xyz[1] + (2.)*math.sqrt(cov[1,1]) #10000 #co[1,1]
    yarr = np.arange(yarrmin,yarrmax,5000)
    zarrmin = 1000 #min(1000, xyz[2] - (3./4.)*co[2,2])
    zarrmax = xyz[2] + (2.)*math.sqrt(cov[2,2]) #10000 #co[2,2]
    zarr = np.arange(zarrmin,zarrmax,5000)
    X, Y, Z = np.meshgrid(xarr,yarr,zarr)
    xyz_list = np.vstack(np.hstack(np.stack((X,Y,Z),axis=-1)))
    xyz_ind=np.arange(0,len(xyz_list),1)
    xyz_prob_pdf = scipy.stats.multivariate_normal.pdf(xyz_list,mean=mean, cov=cov)
    #xyz_prob_cum = scipy.stats.multivariate_normal.cdf(xyz_grid,mean=mean, cov=cov)
    xyz_prob_pdf_norm = xyz_prob_pdf/np.sum(xyz_prob_pdf)
    
    ### Angles PDF ###
    ##################
    # Inside the original code the strike/dip/rake
    # do not depend of the magnitude and position
    bs2_pos = len(pre_selection['BS2_Position_Selection_inn'])
    lat_val=hx ## event position
    lon_val=hy ## event position

    # Conversion into Lat/Lon degree values
    if (lat_val>2000000) or (lat_val<0):
        lat_val = 100001
        lon_val = 100001
        latlon = utm.to_latlon(lat_val,
                               lon_val,
                               ee['ee_utm'][2], northern=True)
    elif (lat_val<1000000) and (lat_val>100000):
        latlon = utm.to_latlon(lat_val,
                               lon_val,
                               ee['ee_utm'][2], northern=True)#ee['ee_utm'][3])
    elif (lat_val>1000000):
        lat_val=100000+(lat_val-999999)
        latlon = utm.to_latlon(lat_val,
                               lon_val,
                               ee['ee_utm'][2]+1, northern=True)#ee['ee_utm'][3])
    else:
        lat_val=999999-(100000-lat_val)
        latlon = utm.to_latlon(lat_val,
                               lon_val,
                               ee['ee_utm'][2]-1, northern=True)#ee['ee_utm'][3])

    #print(latlon[0],hx,latlon[1],hy)
    lat_val=latlon[1]
    lon_val=latlon[0]
    d_latlon=np.zeros((bs2_pos,2))
    d_diff=np.zeros((bs2_pos))
    for val in range(bs2_pos):
            tmp_idx = pre_selection['BS2_Position_Selection_inn'][val]
            d_latlon[val,:] = Discretizations['BS-2_Position']['Val'][tmp_idx].split()
            d_diff[val] = haversine(lat_val, lon_val, d_latlon[val,0], d_latlon[val,1])
    ipos_idx = int(np.argmin(d_diff))
    ipos = pre_selection['BS2_Position_Selection_inn'][ipos_idx]
    ireg = Discretizations['BS-2_Position']['Region'][ipos]       
    # Faccio il load della regione se non già fatto
    if(ireg not in regions_nr):
        #print("...............................", region_files)
        region_info = load_region_infos(ireg         = ireg,
                                        region_info  = region_info,
                                        region_files = region_files)
        regions_nr.append(ireg)
    RegMeanProb_BS4 = region_info[ireg]['BS4_FocMech_MeanProb_valNorm']

    # Non credo che qui ci saranno errori, nel senso che i npy sono creati a partire dai mat conteneti roba
    if(RegMeanProb_BS4.size == 0):
         print(' --> WARNING: region info %d is empty!!!' % (ireg) )
    ipos_reg = np.where(region_info[ireg]['BS4_FocMech_iPosInRegion'] == ipos+1)[1]
    ang_prob_pdf = RegMeanProb_BS4[ipos_reg[0]]
    ang_list = Discretizations['BS-4_FocalMechanism']['Val']
    ang_ind=np.arange(0,len(ang_list),1)
    ang_prob_pdf_norm = ang_prob_pdf/np.sum(ang_prob_pdf)
    print('Ang prob ok')

    ### Combination of the three PDF functions ###
    ##############################################

    M, P, A = np.meshgrid(mag_val_grid,xyz_ind,ang_ind)
    mxyza_list = np.vstack(np.hstack(np.stack((M,P,A),axis=-1)))
    PX, PY, PZ = np.meshgrid(mag_prob_pdf_norm,xyz_prob_pdf_norm,ang_prob_pdf_norm)
    mxyza_prob = np.vstack(np.hstack(np.stack((PX,PY,PZ),axis=-1)))
    mxyza_prob_prod = np.prod(mxyza_prob, axis=1)
    
    ### Creation of the array of cumulated probability intervals associated to the initial ensemble ###
    mxyza_prob_cum = np.zeros(len(mxyza_prob_prod))
    prob_cum = 0
    for i in range(len(mxyza_prob_prod)):
        prob_cum=prob_cum+mxyza_prob_prod[i]
        mxyza_prob_cum[i]= prob_cum
    
    sampler = scipy.stats.qmc.LatinHypercube(d=1)
    random_value = sampler.random(n=Num_Samp)
    para_scen_lhs = np.zeros((Num_Samp,8))
    itmp = 0
    
    for i in random_value:
        ### Each value is associated to a scenario that can be retrieved from the cumulative probability function
        idx,proba = find_nearest(mxyza_prob_cum,i)
        print('find nearest done',itmp)
        para_scen_lhs[itmp,0] = mxyza_list[idx,0] #mag
        XYZ_idx = int(mxyza_list[idx,1])
        ANG_idx = int(mxyza_list[idx,2])
        para_scen_lhs[itmp,1] = xyz_list[XYZ_idx,0] #lat
        para_scen_lhs[itmp,2] = xyz_list[XYZ_idx,1] #lon
        para_scen_lhs[itmp,3] = xyz_list[XYZ_idx,2] #dep
        Sv,Dv,Rv = Discretizations['BS-4_FocalMechanism']['Val'][ANG_idx].split() 
        para_scen_lhs[itmp,4] = float(Sv)
        para_scen_lhs[itmp,5] = float(Dv)
        para_scen_lhs[itmp,6] = float(Rv)
        para_scen_lhs[itmp,7] = ANG_idx
        itmp=itmp+1

    ### Beginning of the parameter selection for each scenario ###
    ##############################################################
    for i in range(Num_Samp):

       print(i)
       iscenbs=i


       ### Choice of the magnitude ###
       ###############################
       mag_val_disc = 0.0
       mag_val=para_scen_lhs[i,0]
       mag_min_presel=np.ndarray.min(pre_selection['sel_BS_Mag_val'][:])
       if mag_val<mag_min_presel:
           mag_val_disc=mag_min_presel
       else:
           mag_val_disc=mag_val

       ### Compute vertical half_width with respect the magnitude ###
       v_hwidth = correct_BS_vertical_position(mag = mag_val)
       h_hwidth = correct_BS_horizontal_position(mag = mag_val)
       area = (2.0*v_hwidth/(math.sin(math.pi/4)))*(2.0*h_hwidth)
       length = 2.0*h_hwidth
       rig = 30.0e9
       ### Moment ###
       Mo=10**((mag_val+10.7)*(3.0/2.0))*1e-7
       slip=Mo/(area*rig)

       dip_min=math.radians(float(10.0))
       depmax=50.0*1000.0
       depmin=1000.0+(v_hwidth*math.sin(dip_min))/math.sin(math.pi/4)
       lat_val=para_scen_lhs[i,1]
       lon_val=para_scen_lhs[i,2]
       dep_val=para_scen_lhs[i,3]
       if dep_val > depmax:
          dep_val=depmax
       elif dep_val < depmin:
          dep_val=depmin
       
       # Conversion into Lat/Lon degree values
       if (lat_val>2000000) or (lat_val<0):
           lat_val = 100001
           lon_val = 100001
           latlon = utm.to_latlon(lat_val,
                                  lon_val,
                                  ee['ee_utm'][2], northern=True)
       elif (lat_val<1000000) and (lat_val>100000):
           latlon = utm.to_latlon(lat_val,
                                  lon_val,
                                  ee['ee_utm'][2], northern=True)#ee['ee_utm'][3])
       elif (lat_val>1000000):
           lat_val=100000+(lat_val-999999)
           latlon = utm.to_latlon(lat_val,
                                  lon_val,
                                  ee['ee_utm'][2]+1, northern=True)#ee['ee_utm'][3])
       else:
           lat_val=999999-(100000-lat_val)
           latlon = utm.to_latlon(lat_val,
                                  lon_val,
                                  ee['ee_utm'][2]-1, northern=True)#ee['ee_utm'][3])

       ### Choice of the angles ###
       ############################

       # Inside the original code the strike/dip/rake
       # do not depend of the magnitude and position
       lat_val=latlon[1]
       lon_val=latlon[0]
       bs2_pos = len(pre_selection['BS2_Position_Selection_inn'])
       d_latlon=np.zeros((bs2_pos,2))
       d_diff=np.zeros((bs2_pos))
       for val in range(bs2_pos):
               tmp_idx = pre_selection['BS2_Position_Selection_inn'][val]
               d_latlon[val,:] = Discretizations['BS-2_Position']['Val'][tmp_idx].split()
               d_diff[val] = haversine(lat_val, lon_val, d_latlon[val,0], d_latlon[val,1])
       ipos_idx = int(np.argmin(d_diff))
       d_lat_val = d_latlon[ipos_idx,0]
       d_lon_val = d_latlon[ipos_idx,1]
       ipos = pre_selection['BS2_Position_Selection_inn'][ipos_idx]
       ireg = Discretizations['BS-2_Position']['Region'][ipos]
       # Faccio il load della regione se non già fatto
       if(ireg not in regions_nr):
           #print("...............................", region_files)
           region_info = load_region_infos(ireg         = ireg,
                                           region_info  = region_info,
                                           region_files = region_files)
           regions_nr.append(ireg)
       RegMeanProb_BS4 = region_info[ireg]['BS4_FocMech_MeanProb_valNorm']
       # Non credo che qui ci saranno errori, nel senso che i npy sono creati a partire dai mat conteneti roba
       if(RegMeanProb_BS4.size == 0):
            print(' --> WARNING: region info %d is empty!!!' % (ireg) )
       ipos_reg = np.where(region_info[ireg]['BS4_FocMech_iPosInRegion'] == ipos+1)[1]
       iangle = int(para_scen_lhs[i,7])
       proba_angles = RegMeanProb_BS4[ipos_reg[0]][iangle]
       str_val,dip_val,rak_val = Discretizations['BS-4_FocalMechanism']['Val'][iangle].split()
       dip_val_rad=math.radians(float(dip_val))

       ### Choice of the angle for the real sampling ###
       #################################################
       # Strike angle: uniform law #
       low_str=max(0,float(str_val)-22.5)
       high_str=min(360,float(str_val)+22.5)
       str_real_val=np.random.uniform(low=low_str, high=high_str)
       # Rake angle: uniform law #
       low_rak=max(0,float(rak_val)-45.0)
       high_rak=min(360,float(rak_val)+45.0)
       rak_real_val=np.random.uniform(low=low_rak, high=high_rak)
       # Dip angle: uniform law #
       low_dip=max(0,float(dip_val)-45.0)
       high_dip=min(90,float(dip_val)+45.0)
       dip_real_val=np.random.uniform(low=low_dip, high=high_dip)
       dip_real_val_rad=math.radians(float(dip_real_val))

       max_dip=max(float(dip_val),dip_real_val)
       max_dip_rad=math.radians(float(max_dip))
       lim_dep=1000.0+(v_hwidth*math.sin(max_dip_rad))/math.sin(math.pi/4)
       if dep_val<lim_dep:
          dep_val=lim_dep


       ### Slip, mag and region ###
       ############################

       mag_list=np.array(Discretizations['BS-1_Magnitude']['Val'][:])
       imag = np.argmin(np.abs(mag_list-mag_val_disc))
       # Depth of the event
       idep = np.argmin(np.abs(Discretizations['BS-3_Depth']['ValVec'][imag][ipos]-(dep_val-(v_hwidth*math.sin(dip_val_rad)/math.sin(math.pi/4)))/1000.0))#/math.sin(math.pi/4)))/1000.0))
       are_val = Discretizations['BS-5_Area']['ValArea'][ireg, imag, iangle]
       len_val = Discretizations['BS-5_Area']['ValLen'][ireg, imag, iangle]
       sli_val = Discretizations['BS-6_Slip']['Val'][ireg, imag, iangle]

       ### Real sampled parameters ###
       ###############################
       real_val[i,0]=ireg
       real_val[i,1]=mag_val
       real_val[i,2]=lat_val
       real_val[i,3]=lon_val
       #real_val[i,4]=dep_val/1000.0 #(dep_val-(v_hwidth*math.sin(dip_val_rad)/math.sin(math.pi/4)))/1000.0 #math.sin(math.pi/4)))/1000.0
       real_val[i,4]=(dep_val-(v_hwidth*math.sin(dip_val_rad)/math.sin(math.pi/4)))/1000.0 #math.sin(math.pi/4)))/1000.0
       real_val[i,5]=str_real_val
       real_val[i,6]=dip_real_val
       real_val[i,7]=rak_real_val
       real_val[i,8]=area/(1000.0**2)
       real_val[i,9]=length/1000.0
       real_val[i,10]=slip
       ##############################
       #### Identification of the corresponding discretized parameters ###
       ###################################################################
       temp_val=np.zeros((10))
       temp_val[0]=ireg
       temp_val[1]=Discretizations['BS-1_Magnitude']['Val'][imag] #pre_selection['sel_BS_Mag_val'][imag] #Discretizations['BS-1_Magnitude']['Val'][imag]
       temp_val[2]= d_lat_val #d_latlon[ipos_idx,0]
       temp_val[3]= d_lon_val #d_latlon[ipos_idx,1]
       temp_val[4]=Discretizations['BS-3_Depth']['ValVec'][imag][ipos][idep]
       temp_val[5]=str_val
       temp_val[6]=dip_val
       temp_val[7]=rak_val
       
       ### Real values ###
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,0]=real_val[i,0] #Reg
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,1]=real_val[i,1] #Mag
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,2]=real_val[i,2] #Lon
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,3]=real_val[i,3] #Lat
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,4]=real_val[i,4] #Depth
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,5]=real_val[i,5] #Strike
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,6]=real_val[i,6] #Dip
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,7]=real_val[i,7] #Rake
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,8]=real_val[i,8] #Area 
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,9]=real_val[i,9] #Length
       samp_ens[Nid]['real_par_scenarios_bs'][iscenbs,10]=real_val[i,10] #Slip

    samp_ens[Nid]['nr_bs_scenarios'] = np.shape(samp_ens[Nid]['prob_scenarios_bs_fact'])[0]
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
    NPS                = kwargs.get('NPS', None)
    Nid                = kwargs.get('Nid', None)
    Num_Samp           = kwargs.get('Num_Samp', None)
    ee                 = kwargs.get('ee', None)
    region_info        = dict()

    samp_ens[Nid]['PScomputedYN'] == False
    
    if(samp_ens[Nid]['PScomputedYN'] == False):# or short_term['PS_computed_YN'] == False):

        #print("--------uuuuuuuuuuuu------------>>", samp_ens[Nid]['PScomputedYN'], short_term['PS_computed_YN'])

        #fbfix 2021-11-26
        samp_ens[Nid]['PScomputedYN']    = False
        #short_term['PS_computed_YN']   = False
        #
        samp_ens[Nid]['nr_ps_scenarios'] = 0

        return samp_ens

    ##### Initiation of the array #####
    Mw=ee['mag']
    sig=ee['MagSigma']
    mag=[]
    num_mag_ps=Num_Samp
    magvaltmp=np.sort(np.random.normal(loc=Mw, scale=sig, size=num_mag_ps))
    magval=magvaltmp
    discr=np.unique(prob_scenes['par_scenarios_ps'][:,1])
    for i in range(len(magvaltmp)):
       #closestmag=min(Discretizations['PS-1_Magnitude']['Val'], key=lambda x:abs(x-magvaltmp[i]))
       closestmag=min(discr, key=lambda x:abs(x-magvaltmp[i]))
       magval[i]=closestmag
    magvalu,magvali = np.unique(magval,return_counts=True)

    #### Correction of the probability ###
    a     = magvalu[0:-1]
    b     = magvalu[1:]
    c     = np.add(a, b) * 0.5
    lower = np.insert(c, 0, -np.inf)
    upper = np.insert(c, c.size, np.inf)
    lower_probility  = norm.cdf(lower, ee['mag_percentiles']['p50'], ee['MagSigma'])
    upper_probility  = norm.cdf(upper, ee['mag_percentiles']['p50'], ee['MagSigma'])
    corr_mag=np.subtract(upper_probility, lower_probility)
    #corr_rake=np.array(weights)

    ### Identification of all the scenario in the initial ensemble that have the corresponding mag and rake values
    random_value = np.random.random(NPS)
    iscenps=0
    vec                      = np.array([100000000])#,100,1,0.0100,1.0000e-04,1.0000e-06])
    arra                     = prob_scenes['par_scenarios_ps']
    par_matrix               = np.transpose(arra[:,1])
    convolved_par_ps         = par_matrix
    for k in range(len(samp_ens[Nid]['iscenps'])):
       scene_matrix             = np.transpose(magval[k])
       convolved_sce_ps         = magval[k] #np.array(vec.dot(scene_matrix)).astype(int)
       [Iloc,nr_scenarios]      = ismember(convolved_par_ps,convolved_sce_ps)
       idx_true_scenarios       = np.where(Iloc)[0]
       if len(idx_true_scenarios):
          int_ens_tmp = np.zeros((len(idx_true_scenarios)))
          prob_cum=0
          icon=0
          tot_prob_cum=np.sum(prob_scenes['ProbScenPS'][idx_true_scenarios])
          for icum in idx_true_scenarios:
             prob_cum=prob_cum+prob_scenes['ProbScenPS'][icum]/tot_prob_cum
             int_ens_tmp[icon]= prob_cum
             icon=icon+1
          int_ens=int_ens_tmp
          random_value = np.random.random()
          idx_tmp,proba=find_nearest(int_ens,random_value)
          idx=idx_true_scenarios[idx_tmp]
       else:
          print('Issue!')
          print(magval[k])
          idx=1

       ### The final probability needs to be corrected from the biased choice of the parameters
       ### The corrections follow the same laws used to select the parameters 
       ### (gaussian for the mag and specific distribution for the rake)
       idx_mag=np.where(abs(magvalu-magval[k])<0.0001)[0]
       #idx_rake=np.where(abs(rakevalu-mag_rake[k,1])<0.0001)[0]
       corr_mag_k=corr_mag[idx_mag]
       #corr_rake_k=corr_rake[idx_rake]
       corr_vec=[corr_mag_k,1.0,1.0,1.0,1.0]

       ### The final probability corresponds to :
       ### - tot_prob_cum/np.sum(prob_scenes['ProbScenBS'] : thei probability associated to the set of selected scenario
       ### - 1/(len(idx_true_scenarios)) : the uniform value of probability from the monte-carlo sampling
       ### (The Monte Carlo sampling has a weight on the results by the dupplicated scenarios that are counted later in the ensemble)
       ### - divided by the correction of the corr_mag and corr_rake
       samp_ens[Nid]['iscenps'][iscenps]=idx
       samp_ens[Nid]['prob_scenarios_ps'][iscenps]=tot_prob_cum/(np.sum(prob_scenes['ProbScenPS'])*len(idx_true_scenarios)*corr_mag_k) #prob_scenes['ProbScenPS'][idx]/(corr_mag_k*corr_rake_k)
       for j in range(5):
           samp_ens[Nid]['prob_scenarios_ps_fact'][iscenps,j]=np.sum(prob_scenes['prob_scenarios_ps_fact'][idx_true_scenarios,j])/(np.sum(prob_scenes['prob_scenarios_ps_fact'][:,j]*len(idx_true_scenarios)*corr_vec[j]))#prob_scenes['prob_scenarios_ps_fact'][idx,j]/(corr_vec[j])
       for j in range(7):
           samp_ens[Nid]['par_scenarios_ps'][iscenps,j]=prob_scenes['par_scenarios_ps'][idx,j]
       iscenps=iscenps+1

    samp_ens[Nid]['nr_ps_scenarios'] = np.shape(samp_ens[Nid]['prob_scenarios_ps_fact'])[0]

    return samp_ens


def load_region_infos(**kwargs):

    ireg        = kwargs.get('ireg', None)
    files       = kwargs.get('region_files', None)
    region_info = kwargs.get('region_info', None)

    info = np.load(files['ModelsProb_Region_files'][ireg-1], allow_pickle=True).item()

    region_info[ireg] = info

    return region_info

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km
