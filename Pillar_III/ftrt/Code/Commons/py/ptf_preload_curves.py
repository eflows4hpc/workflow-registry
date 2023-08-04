import os
import glob
import h5py
from pymatreader import read_mat
import numpy as np
import xarray as xr

def reallocate_curves(**kwargs):

    Config    = kwargs.get('cfg', None)
    args      = kwargs.get('args', None)
    c_files   = kwargs.get('curve_files', None)
    name      = kwargs.get('name', None)

    #curves_py_folder  =  Config.get('pyptf',  'curves')
    #curves_py_folder  =  Config.get('pyptf',  'curves_gl_16')

    curves_py_folder  =  Config.get('pyptf',  'h_curves')

    list_out = []
    for i in range(0,int(Config.get('ScenariosList', 'nr_regions'))):
        d = "%03d" % (i+1)
        def_name = curves_py_folder + os.sep + name + d + '-empty.hdf5'
        #def_name = curves_py_folder + os.sep + name + d + '-empty.npy'
        list_out.append(def_name)

    for i in range(len(c_files)):
        ref_nr = int(c_files[i].split(name)[-1][0:3])
        list_out[ref_nr-1] = c_files[i]

    return list_out

def reallocate_curves_sim(**kwargs):

    Config    = kwargs.get('cfg', None)
    args      = kwargs.get('args', None)
    c_files   = kwargs.get('curve_files', None)
    name      = kwargs.get('name', None)
    ptf_out   = kwargs.get('ptf_out', None)
    POIs      = kwargs.get('POIs', None) 
    sim_pois   = kwargs.get('sim_pois', None)
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

    MC_samp=int(Config.get('Sampling','MC_samp_scen'))

    if MC_samp<1:
       par_reg = ptf_out['new_ensemble_RS'][0]['real_par_scenarios_bs'][:,0]
    else:
       par_reg = ptf_out['new_ensemble_MC'][0]['par_scenarios_bs'][:,0] 


    SIMPOIs = {} #=dict()
    simlon = []
    simlat = []
    with open(sim_pois) as f:
         next(f)
         for lines in f:
             foe = lines.rstrip().split(' ')
             simlon.append(float(foe[0]))
             simlat.append(float(foe[1]))
    tmppois = {'lon': simlon, 'lat': simlat}
    SIMPOIs.update(tmppois)

    final_idx = []
    for ip in range(len(POIs['selected_pois'])):
        VAL_LON=np.abs(np.array(SIMPOIs['lon'])-lon[ip])
        VAL_LAT=np.abs(np.array(SIMPOIs['lat'])-lat[ip])
        VAL=np.add(VAL_LON,VAL_LAT)
        final_idx.append(int(np.argmin(VAL)))

    list_out = {}
    list_out = c_files[:,final_idx]

    #for i in range(0,int(Config.get('ScenariosList', 'nr_regions'))):
    #    d = "%03d" % (i+1)
    #    list_out[d]={}

    #for i in range(len(par_reg)):
    #    ref_nr = int(par_reg[i])
    #    indx = np.argwhere(par_reg==ref_nr)
    #    d = "%03d" % (ref_nr-1)
    #    list_out[d] = c_files[indx,final_idx]

    return list_out


def load_hazard_values(**kwargs):

    Config    = kwargs.get('cfg', None)
    args      = kwargs.get('args', None)
    in_memory = kwargs.get('in_memory', False)
    ptf_out   = kwargs.get('ptf_out', None)
    step2_mod = kwargs.get('step2_mod', None)    
    POIs      = kwargs.get('POIs', None)
    sim_pois  = kwargs.get('sim_pois', None)
    """
    Preload from mat file very time consuming, about 10 hours, so args.preload disabled
    for this section. Only args.preload_scenarios
    """

    if step2_mod == "SIM":

         curves_py_folder  =  Config.get('pyptf',  'h_curves_sim')
         data = xr.open_dataset(curves_py_folder+"/Step2_BS_hmax.nc")
         py_gl_bs_curves_tmp = np.array(data['ts_p2t_gl'].values)
         #py_gl_bs_curves_tmp = np.array(data['ts_max_gl'].values)
         py_gl_bs_curves = reallocate_curves_sim(curve_files=py_gl_bs_curves_tmp, args=args, cfg=Config,ptf_out=ptf_out,POIs=POIs, sim_pois=sim_pois)
         hazard_curves_files = dict()
         hazard_curves_files['gl_bs'] = py_gl_bs_curves

    elif step2_mod == "TSUNMAPS":

         # inizialize empty dict, containing the list of the ps and bs scenarios
         scenarios    = dict()

         # Load scenarios path
         # for hdf5
         #curves_py_folder  =  Config.get('pyptf',  'curves')
         # for npy
         #curves_py_folder  =  Config.get('pyptf',  'curves_gl_16')
         curves_py_folder  =  Config.get('pyptf',  'h_curves')
         #curves_mat_folder =  Config.get('tsumaps','curves')


         # Load ps and bs in pypath
         py_os_ps_curves = sorted(glob.glob(curves_py_folder + os.sep + Config.get('pyptf','os_ps_curves_file_names') + '*'))
         py_os_bs_curves = sorted(glob.glob(curves_py_folder + os.sep + Config.get('pyptf','os_bs_curves_file_names') + '*'))
         py_af_ps_curves = sorted(glob.glob(curves_py_folder + os.sep + Config.get('pyptf','af_ps_curves_file_names') + '*'))
         py_af_bs_curves = sorted(glob.glob(curves_py_folder + os.sep + Config.get('pyptf','af_bs_curves_file_names') + '*'))
         py_gl_ps_curves = sorted(glob.glob(curves_py_folder + os.sep + Config.get('pyptf','gl_ps_curves_file_names') + '*'))
         py_gl_bs_curves = sorted(glob.glob(curves_py_folder + os.sep + Config.get('pyptf','gl_bs_curves_file_names') + '*'))

         py_os_ps_curves = reallocate_curves(curve_files=py_os_ps_curves, args=args, cfg=Config, name=Config.get('pyptf','os_ps_curves_file_names'))
         py_os_bs_curves = reallocate_curves(curve_files=py_os_bs_curves, args=args, cfg=Config, name=Config.get('pyptf','os_bs_curves_file_names'))
         py_af_ps_curves = reallocate_curves(curve_files=py_af_ps_curves, args=args, cfg=Config, name=Config.get('pyptf','af_ps_curves_file_names'))
         py_af_bs_curves = reallocate_curves(curve_files=py_af_bs_curves, args=args, cfg=Config, name=Config.get('pyptf','af_bs_curves_file_names'))
         py_gl_ps_curves = reallocate_curves(curve_files=py_gl_ps_curves, args=args, cfg=Config, name=Config.get('pyptf','gl_ps_curves_file_names'))
         py_gl_bs_curves = reallocate_curves(curve_files=py_gl_bs_curves, args=args, cfg=Config, name=Config.get('pyptf','gl_bs_curves_file_names'))

         hazard_curves_files['gl_ps'] = py_gl_ps_curves
         hazard_curves_files['gl_bs'] = py_gl_bs_curves
         hazard_curves_files['os_ps'] = py_os_ps_curves
         hazard_curves_files['os_bs'] = py_os_bs_curves
         hazard_curves_files['af_ps'] = py_af_ps_curves
         hazard_curves_files['af_bs'] = py_af_bs_curves


    return hazard_curves_files

def mat_curves_to_py_curves(**kwargs):

    curves_py_folder  = kwargs.get('py_path', None)
    curves_mat_folder = kwargs.get('mat_path', None)
    files             = kwargs.get('files', None)
    vmat              = kwargs.get('vmat', None)
    mat_key           = kwargs.get('mat_key', None)

    npy_curves_files = []

    for i in range(len(files)):

        npy_file = files[i].replace(curves_mat_folder,curves_py_folder).replace('.mat','.hdf5')

        #if (os.path.isfile(npy_file)):
        #    continue
        print('   ', npy_file, ' <--- ', files[i])
        py_dict  = read_mat(files[i])
        hf = h5py.File(npy_file, 'w')
        hf.create_dataset(mat_key, data=py_dict[mat_key])
        hf.close()
        #potrebbe aiutare?
        #del(hf)

        #"""
        #key      = [*py_dict][0]
        #np.save(npy_file, py_dict[mat_key]) #, allow_pickle=True)
        #"""
        npy_curves_files.append(npy_file)

        #print(py_dict['osVal_PS'][0:3][:])
        #print(type(py_dict['osVal_PS']))
        #sys.exit()

        """
        try:
            py_dict = read_mat(files[i])
        except:
            continue

        try:
            #np.save(npy_file, py_dict, allow_pickle=True)
            done = save_dict(npy=npy_file, dict=py_dict, cfg=Config)
        except:
            continue

        npy_scenarios_files.append(npy_file)
        print('  --> OK: ', npy_file)
        """
    #print(npy_curves_files)


    return npy_curves_files
