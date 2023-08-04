import math
import numpy as np
from ptf_hazard_curves           import define_thresholds
from scipy.stats import expon

####################################
# Begin
####################################

def compute_ensemble_mare(**kwargs):

    Config                = kwargs.get('cfg', None)
    ee                    = kwargs.get('event_parameters', None)
    args                  = kwargs.get('args', None)
    LongTermInfo          = kwargs.get('LongTermInfo', None)
    POIs                  = kwargs.get('pois', None)
    hazard_curves_files   = kwargs.get('h_curve_files', None)
    ptf_out               = kwargs.get('ptf_out', None)
    type_ens              = kwargs.get('type_ens', None)
    samp_test             = kwargs.get('samp_test', None)
    samp_weight           = kwargs.get('samp_weight', None)
    tg                    = kwargs.get('tsu_data', None) 
    sel_scen              = kwargs.get('list_tmp_scen', None)

    ############################ Mare weight to RS sampling ###################################################
    if type_ens == 'RS':

        for Nid in range(len(ptf_out['new_ensemble_'+type_ens])):
   
            name_type_ens = 'new_ensemble_'+type_ens 
            probability_scenarios=ptf_out[name_type_ens][Nid] 
    
            #### POIs ###
            idx=0
            selection=POIs['selected_pois']
            name=POIs['name']
            lon=np.zeros((len(selection)))
            lat=np.zeros((len(selection)))
            for i in range(len(selection)):
                idx=name.index(selection[i])
                lon[i]=POIs['lon'][idx]
                lat[i]=POIs['lat'][idx]
            
            int_meas                 = np.zeros((len(selection),len(sel_scen)))

            probsel_tmp                  = probability_scenarios['RealProbScenBS'][sel_scen]
            tot_prob                     = np.sum(probsel_tmp)
            probsel                      = probsel_tmp/tot_prob
        
            # Open and read hdf5-curves file
            infile = hazard_curves_files['gl_bs']
            data = np.transpose(infile)
 
            hazards = define_thresholds(cfg = Config, LongTermInfo = LongTermInfo, args = args, pois=POIs)
            hazard_curve_threshold = hazards['original_hazard_curve_threshold'].reshape(1, len(hazards['original_hazard_curve_threshold']))
            #nr_scenarios = isel[0]
            #IIM = data#[:, nr_scenarios]
            int_meas = data
            #for id_tmp in range(len(sel_scen)):
            #   id_sel=sel_scen[id_tmp]
            #   int_meas[:,id_sel]=IIM[:,id_tmp]

            ### Selection of the POIs and Hobs/Hmod comparison ###
            # Selection of the closest POI to each observation location
            nbp=1
            tg_poi = {}
            tg_poi['lat'] = np.zeros((len(tg),nbp))
            tg_poi['lon'] = np.zeros((len(tg),nbp))
            tg_poi['id'] = np.zeros((len(tg),nbp))
            tg_poi['dist'] = np.zeros((len(tg),nbp))
            tg_poi['wei'] = np.zeros((len(tg),nbp))
            tg_poi['ip'] = np.zeros((len(tg),nbp))
            name=POIs['name']
            selection=POIs['selected_pois']
            print('selection',len(selection))
            for tg_id in range(len(tg)):
                tglon=tg[tg_id,0]
                tglat=tg[tg_id,1]
                dist=np.ones((len(selection)))
                index=np.ones((len(selection)))
                for poi_id in range(len(selection)):
                    idxp=name.index(selection[poi_id])
                    lon=POIs['lon'][idxp]
                    lat=POIs['lat'][idxp]
                    dist[poi_id]=math.sqrt((tglon-lon)**2+(tglat-lat)**2)*111.0
                    index[poi_id]=idxp
                for ip in range(nbp):
                    idx = np.argmin(dist)
                    idxp=name.index(selection[idx])
                    tg_poi['id'][tg_id,ip] = idxp
                    tg_poi['lon'][tg_id,ip] = POIs['lon'][idxp]
                    tg_poi['lat'][tg_id,ip] = POIs['lat'][idxp]
                    tg_poi['dist'][tg_id,ip] = dist[idx]
                    tg_poi['ip'][tg_id,ip] = idx
                    dist[idx]=100000
                for ip in range(nbp):
                    tmp = (1.0/tg_poi['dist'][tg_id,ip])/np.sum(1.0/tg_poi['dist'][tg_id])
                    tg_poi['wei'][tg_id,ip] = tmp # tg_poi['dist'][tg_id,i]/np.sum(tg_poi['dist'][tg_id])
 
            # For each POI: selection of the maximal observation 
            tg_id=0
            while tg_id < len(tg):
                arr = tg_poi['id'][:,0]
                idxp = tg_poi['id'][tg_id,0]
                count = np.count_nonzero(arr == idxp)
                if count>1:
                   where_arr = np.where(arr == idxp)
                   position=where_arr[0]
                   valu_max = np.argmax(tg[position,2])
                   position = np.delete(position,valu_max)
                   tg = np.delete(tg,position,axis=0)
                   tg_poi['id'] = np.delete(tg_poi['id'],position,axis=0)
                   tg_poi['ip'] = np.delete(tg_poi['ip'],position,axis=0)
                   tg_poi['lon'] = np.delete(tg_poi['lon'],position,axis=0)
                   tg_poi['lat'] = np.delete(tg_poi['lat'],position,axis=0)
                   tg_poi['dist'] = np.delete(tg_poi['dist'],position,axis=0)
                   tg_poi['wei'] = np.delete(tg_poi['wei'],position,axis=0)
                tg_id += 1
    
            # Dividing the tide-gage and run-up value by 2
            origin = tg[:,2]/2.0
            omean=np.mean(origin)

            # Initialisation of the obs-mod comparison dict
            #ptf_out[name_type_ens][Nid]['all_ptf_val']=np.zeros((len(int_meas[0]),len((int_meas))))
            ptf_out[name_type_ens][Nid]['ptf_val']=np.zeros((len(int_meas[0]),len(tg)))
            ptf_out[name_type_ens][Nid]['obs_val']=np.zeros((len(int_meas[0]),len(tg)))
            ptf_out[name_type_ens][Nid]['ptf_obs_diff']=np.zeros((len(int_meas[0]),len(tg)))
            ptf_out[name_type_ens][Nid]['ptf_obs_norm']=np.zeros((len(int_meas[0]),len(tg)))
            ptf_out[name_type_ens][Nid]['NRMS']=np.zeros((len(int_meas[0])))            
            ptf_out[name_type_ens][Nid]['tg_pois']=tg_poi
            ptf_out[name_type_ens][Nid]['tg_real']=tg
            final_logk=np.zeros(len(int_meas[0]))
            final_logK=np.zeros(len(int_meas[0]))
            NRMS=np.zeros(len(int_meas[0]))
            NRMSE=np.zeros(len(int_meas[0]))
            n=float(len(tg))            
            sigmasimu=1.0            
   
 
            for i in range(len(int_meas[0])): #scenario
                logk=0.0
                logK=0.0
                nrmsd=0.0
                nrmso=0.0
                #for all_obs in range(len(int_meas)):
                #    print('entered the ptf val', int_meas[all_obs,i])
                #    ptf_out[name_type_ens][Nid]['all_ptf_val'][i,all_obs]=int_meas[all_obs,i]
                for j in range(len(tg)): #pois
                    datj=0.0
                    jp=int(tg_poi['ip'][j,0])
                    datj=int_meas[jp,i] #np.exp(np.log(int_meas[jp,i])+(sigmasimu**2)/2.0)
                    #for jjp in range(nbp):
                    #    jp=int(tg_poi['ip'][j,jjp])
                    #    datj=datj+(int_meas[jp,i]*tg_poi['wei'][j,jjp])
                    orig=origin[j]
                    if (datj<0.0001):
                        datj=0.0001
                    if (orig<0.0001):
                        orig=0.0001
                    #Correction of the median modelled value to the mean one
                    datj=np.exp(np.log(datj)+(sigmasimu**2)/2.0)
                    #logk=logk+(np.log10(orig/datj))**2
                    logK=logK+np.log10(orig/datj)
                    nrmsd=nrmsd+(orig-datj)**2
                    nrmso=nrmso+(orig-omean)**2
                    ptf_out[name_type_ens][Nid]['ptf_val'][i,j]=datj
                    ptf_out[name_type_ens][Nid]['obs_val'][i,j]=orig
                    ptf_out[name_type_ens][Nid]['ptf_obs_diff'][i,j]=datj-orig
                    #print('Observations !!! ',name_type_ens)
                    if orig>0.1:
                       ptf_out[name_type_ens][Nid]['ptf_obs_norm'][i,j]=(datj-orig)/orig
                    else:
                       ptf_out[name_type_ens][Nid]['ptf_obs_norm'][i,j]=-9999
                
                ### Choice of the metric to be used for reweighting ###
                #final_logk[i]=math.sqrt(((1.0/n)*logk-((1.0/n)*(logK))**2))
                final_logK[i]=10**((1.0/n)*logK)
                NRMS[i]=math.sqrt(nrmsd)/math.sqrt(nrmso)
                #NRMSE[i]=math.sqrt(nrmsd/len(origin))/np.amax(origin) 
                ptf_out[name_type_ens][Nid]['NRMS'][i]=NRMS[i]
                #ptf_out[name_type_ens][Nid]['NRMS'][i]=final_logK[i]                     
    
            mare_weight_1=np.zeros((len(final_logK)))
            #mare_weight_1=expon.pdf(NRMS,loc=0,scale=1)
            mare_weight_1=expon.pdf(NRMS,loc=0,scale=1.0)
            mare_weight=mare_weight_1/np.sum(mare_weight_1)
            
            ptf_out[name_type_ens][Nid]['mare_proba']=mare_weight
            ProbScenBS_all=ptf_out[name_type_ens][Nid]['RealProbScenBS'][sel_scen]
            ProbScenBS_temp=ProbScenBS_all*mare_weight
            ptf_out[name_type_ens][Nid]['RealProbScenBS']=ProbScenBS_temp/np.sum(ProbScenBS_temp)

    return ptf_out


def find_nearest(array, value):
    arr = np.asarray(array)
    idx = 0
    diff = arr-value
    diff[diff<1e-26]=100.0
    idx=diff.argmin()
    return idx,array[idx]

