#!/usr/bin/env python

# Import system modules
import os
import ast
import sys
import glob
import h5py
import numpy as np

from pymatreader         import read_mat
from mat4py              import loadmat
from shapely             import geometry

from ptf_sptha_utilities import countX, region_type_number_2_array, region_coordinates_points_splitter_2_dictionary
from ptf_sptha_utilities import get_BS_discretizations, get_PS_discretizations


def load_PSBarInfo(**kwargs):

    Config = kwargs.get('cfg', None)
    args   = kwargs.get('args', None)
    
    PSBarInfo_py   = Config.get('pyptf','PSBarInfo') 
    PSBarInfo_Dict = np.load(PSBarInfo_py, allow_pickle=True).item()

    return PSBarInfo_Dict



def check_if_path_exists(**kwargs):

    path      = kwargs.get('path', None)
    action    = bool(kwargs.get('path', False))

    if os.path.isdir(path):
        return True

    if os.path.isdir(path) == False:
        if action == True:
            os.mkdir(path)
        else:
            return False

    return True

def save_dict(**kwargs):

    Config    = kwargs.get('cfg', None)
    py_dict   = kwargs.get('dict', None)
    npy_file  = kwargs.get('npy', None)
    type_XS   = kwargs.get('type_XS', None)

    all_dict  = Config.get('ScenariosList','BS_all_dict')
    nr_coll   = int(Config.get('ScenariosList','BS_parameter_nr_coll'))


    if(all_dict == 'Yes'):
        np.savez_compressed(npy_file, py_dict)
    else:
        if(nr_coll == -1):
            np.savez_compressed(npy_file, py_dict)
        else:
            try:
                np.savez_compressed(npy_file, py_dict[:,0:nr_coll])
                print("Saved ", npy_file+'.npz')
            except:
                # For empty nparray in py_dict['ScenariosList']['parameter']
                np.savez_compressed(npy_file, py_dict)
                print("Saved empty", npy_file+'.npz')

    #uncompressed npy
    """
    if(all_dict == 'Yes'):
        np.save(npy_file, py_dict, allow_pickle=True)
    else:
        if(nr_coll == -1):
            np.save(npy_file, py_dict['ScenariosList']['parameter'], allow_pickle=True)
        else:
            try:
                np.save(npy_file, py_dict['ScenariosList']['parameter'][:,0:nr_coll], allow_pickle=True)
            except:
                # For empty nparray in py_dict['ScenariosList']['parameter']
                np.save(npy_file, py_dict['ScenariosList']['parameter'], allow_pickle=True)
    """


    return True

def select_pois_and_regions(**kwargs):

    Config          = kwargs.get('cfg', None)
    args            = kwargs.get('args', None)
    POIs            = kwargs.get('pois_dictionary', None)
    regionalization = kwargs.get('regionalization_dictionary', None)

    ele_args = args.pois.split(' ')
    reg_args = args.regions.split(' ')

    tmp = []
    if(args.pois == '-1' or args.pois == 'mediterranean'):
       SelectedPOIs = ['mediterranean']
    #if(RepresentsInt(args.pois) == False and len(ele_args)<2):
    #   SelectedPOIs = [args.pois]
    if(args.pois != '-1' and args.pois != 'mediterranean'):
    #if(len(ele_args) >= 1):
        for i in range(len(ele_args)):
            if(RepresentsInt(ele_args[i]) == True):
                tmp.append(int(ele_args[i]))
            else:
                tmp.append(ele_args[i])
        SelectedPOIs = tmp

    tmp = []
    if(args.regions == '-1' or args.regions == 'all'):
        SelectedRegions = [-1]
    elif(len(reg_args) >= 1):
        for i in range(len(reg_args)):
            tmp.append(int(reg_args[i]))
        SelectedRegions = tmp

    if (args.ignore_regions != None):
        noreg_args = args.ignore_regions.split(' ')
        noreg_args = [ int(x) for x in noreg_args ]
        IgnoreRegions = noreg_args
    else:
        IgnoreRegions = []

    #print(POIs['Mediterranean'])
    #print(POIs['name'][1244])
    #POIs['Mediterranean'][1244] = 1
    #sys.exit()

    # questa parte presa paroparo
    if (len(SelectedPOIs) == 0):
      SelectedPOIs = POIs['name']

    elif (len(SelectedPOIs) == 1 and  SelectedPOIs[0].startswith('mediterranean')):
      tmpmed = np.array(POIs['Mediterranean'])
      tmp = np.nonzero(tmpmed)[0]
      xpoi = SelectedPOIs[0].split('-')

      if (len(xpoi) == 1):
         SelectedPOIs = [POIs['name'][j] for j in tmp] #All
      else:
         step = int(xpoi[1])
         SelectedPOIs = [POIs['name'][j] for j in tmp[::step]] #1 every step

    if (len(SelectedRegions) == 0):
      SelectedRegions = range(regionalization['Npoly'])
      IgnoreRegions = [42]  #!!REGION 43 NOT AVAILABLE!!#
    else:
      IgnoreRegions = []

    POIs['selected_pois']    = SelectedPOIs
    POIs['selected_regions'] = SelectedRegions
    POIs['ignore_regions']   = IgnoreRegions
    POIs['nr_selected_pois'] = len(SelectedPOIs)


    return POIs


def mat_scenarios_to_py_scenarios(**kwargs):

    """
    ScenariosListBS are very large (45GB in mat and 15GB in py). Only a part of thet files are used by ptf:
    ['ScenariosList']                      all         file                      22 sec to read on cat-scenaroi
    ['ScenariosList']['parameter']         the netsed dict used  (5.8GB)          3 sec
    ['ScenariosList']['parameter'][:,0:7]  the part of the netsed dict used  (4.1GB) 1.6

    !! 10 hor about from mat to npy in any case

    config['ScenariosList']                             = {}
    config['ScenariosList']['BS_all_dict']              = 'No' if yes all dict.
    config['ScenariosList']['BS_parameter_nr_coll']     = '7'  if this is -1 is off

    """

    scenarios_py_folder  = kwargs.get('py_path', None)
    scenarios_mat_folder = kwargs.get('mat_path', None)
    files                = kwargs.get('files', None)
    vmat                 = kwargs.get('vmat', None)
    Config               = kwargs.get('cfg', None)
    type_XS              = kwargs.get('type_XS', None)



    npy_scenarios_files = []

    if(vmat == 4):
        for i in range(len(files)):
            print(files[i])
            npy_file = files[i].replace(scenarios_mat_folder,scenarios_py_folder).replace('.mat','.npy')
            try:
                array_scenarios = read_mat(files[i])['ScenarioListBSReg']['Parameters']
            except:
                continue

            try:
                done = save_dict(npy=npy_file, dict=array_scenarios, cfg=Config, type_XS=type_XS)
            except:
                continue


            npy_scenarios_files.append(npy_file+'.npz')
            print('  --> OK: ', npy_file+'.npz')



    else:
        for i in range(len(files)):
            npy_file = files[i].replace(scenarios_mat_folder,scenarios_py_folder).replace('.mat','.npy')

            try:
                py_dict = loadmat(files[i], 'r')
            except:
                continue

            # Remove useless keys:
            del py_dict['ScenarioListPSReg']['modelID']
            del py_dict['ScenarioListPSReg']['ID']
            del py_dict['ScenarioListPSReg']['IDModel']
            del py_dict['ScenarioListPSReg']['barPSInd']

            try:
                #uncompressed
                #np.save(npy_file, py_dict, allow_pickle=True)
                #compressed
                np.savez_compressed(npy_file, **py_dict, allow_pickle=True)

            except:
                continue

            npy_scenarios_files.append(npy_file+'.npz')
            print('  --> OK: ', npy_file+'.npz')



    return npy_scenarios_files

def load_Scenarios_Reg(**kwargs):

    Config    = kwargs.get('cfg', None)
    args      = kwargs.get('args', None)
    list_PS   = kwargs.get('list_PS', None)
    type_XS   = kwargs.get('type_XS', None)
    in_memory = kwargs.get('in_memory', False)

    """
    Preload from mat file very time consuming, about 10 hours, so args.preload disabled
    for this section. Only args.preload_scenarios
    """

    # inizialize empty dict, containing the list of the ps and bs scenarios
    scenarios    = dict()

    # Load scenarios path
    scenarios_py_folder  =  Config.get('pyptf',  'Scenarios_py_Folder')
    scenarios_mat_folder =  Config.get('tsumaps','Scenarios_mat_Folder')

    # Load ps and bs in pypath
    py_ps_scenarios = sorted(glob.glob(scenarios_py_folder + os.sep + 'ScenarioListPS*npz'))
    py_bs_scenarios = sorted(glob.glob(scenarios_py_folder + os.sep + 'ScenarioListBS*npz'))

    if (len(py_ps_scenarios) == 0 or args.preload_scenarios == 'Yes'):

        path_py_exist = check_if_path_exists(path=scenarios_py_folder, create=True)

        print("PreLoad PS Scenarios Files in Folder for npy conversion     <------ ", scenarios_mat_folder)
        mat_ps_scenarios = sorted(glob.glob(scenarios_mat_folder + os.sep + 'ScenarioListPS*'))
        py_ps_scenarios  = mat_scenarios_to_py_scenarios(mat_path = scenarios_mat_folder,
                                                         py_path  = scenarios_py_folder,
                                                         files    = mat_ps_scenarios,
                                                         cfg      = Config,
                                                         type_XS='PS')

    if (len(py_bs_scenarios) == 0 or args.preload_scenarios == 'Yess'):

        path_py_exist = check_if_path_exists(path=scenarios_py_folder, create=True)

        print("PreLoad BS Scenarios Files in Folder for npy conversion     <------ ", scenarios_mat_folder)
        mat_ps_scenarios = sorted(glob.glob(scenarios_mat_folder + os.sep + 'ScenarioListBS*'))
        py_ps_scenarios  = mat_scenarios_to_py_scenarios(mat_path = scenarios_mat_folder,
                                                         py_path  = scenarios_py_folder,
                                                         files    = mat_ps_scenarios,
                                                         cfg      = Config,
                                                         vmat     = 4,
                                                         type_XS='BS')

    # Start loads
    # ps_scenarios[nr_reg][dict]
    # bs_scenarios[nr_reg][np.darray]
    ps_scenarios = load_scenarios(list_scenarios=py_ps_scenarios, type_XS='PS', cfg=Config)
    bs_scenarios = load_scenarios(list_scenarios=py_bs_scenarios, type_XS='BS', cfg=Config)

    return ps_scenarios, bs_scenarios

def load_scenarios(**kwargs):

    list_scenarios  = kwargs.get('list_scenarios', None)
    type_XS         = kwargs.get('type_XS', None)
    Config          = kwargs.get('cfg', None)

    all_dict        = Config.get('ScenariosList','BS_all_dict')

    dic = dict()


    if(type_XS == 'PS'):

        print('Load PS_scenarios in memory' )
        for i in range(len(list_scenarios)):

            """
            # Per quelli vecchi non compressed
            tmp = np.load(list_scenarios[i], allow_pickle=True).item()
            a = int(list_scenarios[i].split('_')[1].replace('Reg',''))
            dic[a] = tmp['ScenarioListPSReg']
            #print(list_scenarios[i], a, dic[a].keys())
            #sys.exit()
            """

            # Nuovo compressed sarebbe
            tmp = np.load(list_scenarios[i], allow_pickle=True)
            a = int(list_scenarios[i].split('_')[1].replace('Reg',''))
            dic[a] = {}
            dic[a]['Parameters']       = tmp['ScenarioListPSReg'].item()['Parameters']
            dic[a]['SlipDistribution'] = tmp['ScenarioListPSReg'].item()['SlipDistribution']
            dic[a]['magPSInd']         = tmp['ScenarioListPSReg'].item()['magPSInd']
            dic[a]['modelVal']         = tmp['ScenarioListPSReg'].item()['modelVal']


    if(type_XS == 'BS'):

        print('Load BS_scenarios in memory' )
        """
        # Questo per vecchi npy immensi
        for i in range(len(list_scenarios)):
            if(all_dict == "Yes"):
                tmp = np.load(list_scenarios[i], allow_pickle=True).item()
            else:
                tmp = np.load(list_scenarios[i], allow_pickle=True)

            a = int(list_scenarios[i].split('_')[1].replace('Reg',''))

            dic[a] = tmp
            #print(tmp)
        """

        # Questo per nuovi npz
        for i in range(len(list_scenarios)):
            if(all_dict == "Yes"):
                tmp = np.load(list_scenarios[i])
            else:
                tmp = np.load(list_scenarios[i])

            a = int(list_scenarios[i].split('_')[1].replace('Reg',''))

            dic[a] = tmp['arr_0']

    print('... loading completed')
    return dic




def load_mesh(**kwargs):

    mesh_path = kwargs.get('path', None)
    mesh_name = kwargs.get('mesh_name', None)
    mesh_face = kwargs.get('mesh_face', None)
    mesh_node = kwargs.get('mesh_node', None)

    mesh_d = dict()
    mesh_d['name'] = mesh_name

    f = os.path.join(mesh_path, mesh_face)
    n = os.path.join(mesh_path, mesh_node)

    mesh_d['faces']          = dict()
    mesh_d['faces']['nr']    = np.loadtxt(f,usecols=0).astype(int) -1
    mesh_d['faces']['n0']    = np.loadtxt(f,usecols=1).astype(int) -1
    mesh_d['faces']['n1']    = np.loadtxt(f,usecols=2).astype(int) -1
    mesh_d['faces']['n2']    = np.loadtxt(f,usecols=3).astype(int) -1

    mesh_d['nodes']          = dict()
    mesh_d['nodes']['nr']    = np.loadtxt(n,usecols=0).astype(int) -1
    mesh_d['nodes']['lon']   = np.loadtxt(n,usecols=1)
    mesh_d['nodes']['lat']   = np.loadtxt(n,usecols=2)
    mesh_d['nodes']['depth'] = np.loadtxt(n,usecols=3)

    return mesh_d

def set_baricenter(**kwargs):

    mesh             = kwargs.get('mesh', None)

    mesh['bari']         = dict()
    mesh['bari']['lat']  = dict()
    mesh['bari']['lon']  = dict()
    mesh['bari']['depth']= dict()

    lat   = np.zeros(len(mesh['faces']['nr']))
    lon   = np.zeros(len(mesh['faces']['nr']))
    depth = np.zeros(len(mesh['faces']['nr']))
    #sys.exit()

    for i in mesh['faces']['nr']:
        n0 = mesh['faces']['n0'][i]
        n1 = mesh['faces']['n1'][i]
        n2 = mesh['faces']['n2'][i]

        ll0 = mesh['nodes']['lat'][n0]
        ll1 = mesh['nodes']['lat'][n1]
        ll2 = mesh['nodes']['lat'][n2]

        lo0 = mesh['nodes']['lon'][n0]
        lo1 = mesh['nodes']['lon'][n1]
        lo2 = mesh['nodes']['lon'][n2]

        dep0 = mesh['nodes']['depth'][n0]
        dep1 = mesh['nodes']['depth'][n1]
        dep2 = mesh['nodes']['depth'][n2]

        data = ((ll0,lo0,dep0), (ll1,lo1,dep1), (ll2,lo2,dep2))
        lat[i], lon[i], depth[i] = np.mean(data, axis=0)
        #mesh['bari']['lat'][i],  mesh['bari']['lon'][i], mesh['bari']['depth'][i]= np.mean(data, axis=0)
        #print(mesh['name'])
        #print(data, lat[i], lon[i], depth[i])
        #sys.exit()
    mesh['bari']['lat']=lat
    mesh['bari']['lon']=lon
    mesh['bari']['depth']=depth

    return mesh


def PSBarInfo_mat2py(**kwargs):


    cfg = kwargs.get('cfg', None)
    args = kwargs.get('args', None)
    matf = kwargs.get('matfile', None)
    npyf = kwargs.get('pyfile', None)


    D = dict()
    f = loadmat(matf)['PSBarInfo']['BarPSperModelDepth']
    x = np.squeeze(f).reshape(16,18)
    test_dict = {}

    for i in range(0,16):
        for j in range(0,18):

            #x[i][j] = np.array(x[i][j])
            #b  = x[i][j]  #.tolist()

            test_dict.setdefault(j, {})[i] = np.array(x[i][j])

    D['BarPSperModelDepth'] =  test_dict


    f = loadmat(matf)['PSBarInfo']['BarPSperModel']
    x = np.squeeze(f).reshape(16,18)

    test_dict = {}

    for i in range(0,16):
        for j in range(0,18):

            x[i][j] = np.array(x[i][j])
            b  = x[i][j].tolist()

            if(len(b)==0):
               xx= np.array([])
               yy= np.array([])
            elif(len(b)==2):
               xx= np.array(b[0])
               yy= np.array(b[1])
            else:
               xx = np.array(b)[:,0]
               yy = np.array(b)[:,1]

            #print("____________",i,j,xx.size)
            loc = {'pos_xx' : xx, 'pos_yy' : yy}
            test_dict.setdefault(j, {})[i] = loc

    D['BarPSperModel'] = test_dict

    f = loadmat(matf)['PSBarInfo']['BarPSperModelReg']
    x = np.squeeze(f).reshape(16,18)

    test_dict = {}

    for i in range(0,16):
        for j in range(0,18):

            #x[i][j] = np.array(x[i][j])
            #b  = x[i][j]  #.tolist()

            test_dict.setdefault(j, {})[i] = np.array(x[i][j])

    D['BarPSperModelReg'] = test_dict



    f = loadmat(matf)['PSBarInfo']['BarPSModelYN']
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = np.array(x[i][j])
    x = np.squeeze(f)
    D['BarPSModelYN'] = x

    return D



def ModelsProb_Region_hdf5_to_py(**kwargs):

    hdf5            = kwargs.get('hdf5', None)
    npy             = kwargs.get('npy', None)
    discretizations = kwargs.get('discretizations', None)
    modelweights    = kwargs.get('modelweights', None)
    ev_dict         = kwargs.get('dictionary', None)
    args            = kwargs.get('args', None)

    if 'ModelsProb_Region_files' not in ev_dict:
        ev_dict['ModelsProb_Region_files'] = []

    if os.path.isfile(npy) == True and args.preload_BS4 != 'Yes':
        ev_dict['ModelsProb_Region_files'].append(npy)
        return ev_dict

    print("Load                                                    <------ ", hdf5)
    f = h5py.File(hdf5, 'r')

    out_dict = dict()

    out_dict['BS_exist']                = np.array(f['ModelsProb_Region']['BS_exist'])
    out_dict['PS_exist']                = np.array(f['ModelsProb_Region']['PS_exist'])
    out_dict['BS4_FocMech_iPosInRegion']= np.array(f['ModelsProb_Region']['BS4_FocMech']['iPosInRegion'])

    # Unused. Removed from npy files after 2022.02.23 fb
    #out_dict['BS1_Mag_lambda']          = np.array(f['ModelsProb_Region']['BS1_Mag']['lambda'])
    #out_dict['BS2_Pos_prob']            = np.array(f['ModelsProb_Region']['BS2_Pos']['prob'])
    #out_dict['BS3_Depth_prob']          = np.array(f['ModelsProb_Region']['BS3_Depth']['prob'])
    #out_dict['BS4_FocMech_prob']        = np.array(f['ModelsProb_Region']['BS4_FocMech']['prob'])
    #out_dict['BS4_FocMech_probNorm']    = np.array(f['ModelsProb_Region']['BS4_FocMech']['probNorm'])

    #fmprob = out_dict['BS4_FocMech_prob']
    fmprob = np.array(f['ModelsProb_Region']['BS4_FocMech']['prob'])

    len_ID   = len(discretizations['BS-4_FocalMechanism']['ID'])
    len_Ipos = len(out_dict['BS4_FocMech_iPosInRegion'][0])
    tmp=np.empty((len_Ipos, len_ID))
    tmpnorm=np.empty((len_Ipos, len_ID))

    for iPos in range(len_Ipos):
        for iAng in range(len_ID):
           allpr = np.squeeze(fmprob[:,iAng,iPos])
           tmp[iPos,iAng] = sum(allpr * modelweights['BS4_FocMech']['Wei'])/sum(modelweights['BS4_FocMech']['Wei'])
        tmpnorm[iPos,:] = tmp[iPos,:]/sum(tmp[iPos,:])

    out_dict['BS4_FocMech_MeanProb_val'] = tmp
    out_dict['BS4_FocMech_MeanProb_valNorm'] = tmpnorm


    np.save(npy, out_dict, allow_pickle=True)
    print("Created ", npy)

    ev_dict['ModelsProb_Region_files'].append(npy)

    return ev_dict

def RepresentsInt(s):
  try:
    int(s)
    return True
  except ValueError:
    return False


def load_lookup_tables_files(**kwargs):

    file  = kwargs.get('file', None)
    tag  = kwargs.get('tag', None)
    tmp = []

    if(tag == 'HazardCurveThresholds' or tag =='InputIntensities'):
        f = open(file,"r")
        for lines in f:
            tmp.append(float(lines))
        f.close()
        tmptable = {tag: tmp}

    elif(tag== 'LookupTable'):
        f = h5py.File(file, 'r')
        LookupTable = f['hcs/value'][()]
        #print(LookupTable.shape)  => (251, 1000, 51) IN MATLAB IT'S [51 1000 251] TO FIX?????
        tmptable = {tag: LookupTable}

    else:
        sys.exit()

    return  tmptable



def load_moho_grid_array_and_depth(**kwargs):

    infile  = kwargs.get('infile', None)
    info    = kwargs.get('info', None)
    lon_pos = kwargs.get('lon_pos', None)
    lat_pos = kwargs.get('lat_pos', None)
    inode   = kwargs.get('inode', None)

    temp     = []
    moho_dep = []
    moho_all = []

    if(inode == 0 or inode == 2 or inode== 3 or inode ==5):
       return temp


    f = open(infile, "r")
    for lines in f:
        foe = lines.rstrip().split()
        temp.append([float(foe[lon_pos]), float(foe[lat_pos])])
        moho_all.append([float(foe[lon_pos]), float(foe[lat_pos]), float(foe[3])])
        moho_dep.append(float(foe[3]))
    f.close()

    return np.array(temp), np.array(moho_all), moho_dep

def load_grid_array(**kwargs):

    infile  = kwargs.get('infile', None)
    info    = kwargs.get('info', None)
    lon_pos = kwargs.get('lon_pos', None)
    lat_pos = kwargs.get('lat_pos', None)
    inode   = kwargs.get('inode', None)

    temp = []

    if(inode == 0 or inode == 2 or inode== 3 or inode ==5):
       return temp

    f = open(infile, "r")
    for lines in f:
        foe = lines.rstrip().split()
        temp.append([float(foe[lon_pos]), float(foe[lat_pos])])

    return np.array(temp)



def load_barycenter_file(**kwargs):

    Config = kwargs.get('cfg', None)

    faults_dir =Config.get('tsumaps','Faults_Folder')
    baricenter_file = os.path.join(faults_dir, Config.get('PS','Barycenters_File'))
    print("Load Barycenters_File %-33s <------ %s" % ('',baricenter_file))
    bary_grid_array = load_grid_array(infile = baricenter_file, info='bary', lon_pos=1, lat_pos=2)
    f = open(baricenter_file,"r")
    bar_dep = []
    bar_pos = []
    for lines in f:
       foe = lines.rstrip().split()
       bar_dep.append(float(foe[3]))
       bar_pos.append((float(foe[1]), float(foe[2])))
    f.close()

    return bary_grid_array, bar_dep, bar_pos







########################################################################################################
#################   Main ptf _preload function    ######################################################
########################################################################################################
def ptf_preload(**kwargs):

    Config = kwargs.get('cfg', None)
    args   = kwargs.get('args', None)

    empty_space = "%64s" % ('')

    # Loading all names of files and folder from configuration files
    project_name = Config.get('Project','Name')
#    INPDIR       = Config.get('Project','pyPTF_Folder')

#    print('Define Folder to store .npy file from tsumaps preload   ------> ', INPDIR)
#    if not os.path.exists(INPDIR):
#       os.makedirs(INPDIR)

    #Regionalization
    regionalization_npy = Config.get('pyptf','Regionalization_npy')
    regionalization_dir = Config.get('tsumaps','Regionalization_Folder')
    regionalization_txt = Config.get('tsumaps','Regionalization_txt')
    file_regions        =  os.path.join(regionalization_dir, regionalization_txt)

    # Pois
    pois_npy            = Config.get('pyptf','POIs_npy')
    pois_dir            = Config.get('tsumaps','POIs_Folder')
    pois_lst            = Config.get('tsumaps','POIs_File')
    file_pois           = os.path.join(pois_dir,pois_lst)
    #poi_dictionary_file = os.path.join(INPDIR,'POIs')

    # Discretization
    #f_dis = Config.get('Project','Discretization_npy')
    discretization_dir = Config.get('tsumaps','Discretization_Folder')
    discretization_npy = Config.get('pyptf','Discretization_npy')
    bs_nodes           = Config.get('BS','EventTreeNodes').split(',')
    ps_nodes           = Config.get('PS','EventTreeNodes').split(',')
    file_moho          = os.path.join(discretization_dir, Config.get('BS','Moho_File'))


    # wheight
    #f_dis = Config.get('Project','Weight_npy')
    #models_dir = Config.get('Project','Weight_Folder')
    weight_npy         = Config.get('pyptf','Weight_npy')
    weight_dir         = Config.get('tsumaps','Weight_Folder')

    #config['tsumaps']['ModelWeight']             = tsumaps_ModelWeight
    #config['pyptf']['ModelWeight']               = pyptf_ModelWeight

    weight_npy   = Config.get('pyptf','ModelWeight')
    weight_mat   = Config.get('tsumaps','ModelWeight')

    # Lookup tables
    lookup_tables_npy  = Config.get('pyptf','HazCond_npy')
    table_dir          = Config.get('tsumaps','HazCond_Folder')
    filename           = Config.get('tsumaps','HazCond_File')
    table_file         = os.path.join(table_dir, filename)
    mih_dir            = Config.get('tsumaps','MIH_Folder')
    filename           = Config.get('tsumaps','MIHthr_File')
    file_mihthr        = os.path.join(mih_dir, filename)
    filename           = Config.get('tsumaps','MIHunc_File')
    file_mihsteps      = os.path.join(mih_dir, filename)

    # slab meshes
    path_mesh          = Config.get('lambda','mesh_path')
    mesh_file_npy      = Config.get('Files','meshes_dictionary')
    faces              = ast.literal_eval(Config.get('lambda','mesh_faces'))
    nodes              = ast.literal_eval(Config.get('lambda','mesh_nodes'))
    names              = ast.literal_eval(Config.get('lambda','mesh_names'))



    ############################################
    ##### Load Meshes of slabs
    # Prendere i dati di mesh da file config


    if os.path.isfile(path_mesh + os.sep + mesh_file_npy) and args.preload_mesh == 'No':
        print('Load dictionary for slab meshes:                        <------', path_mesh + os.sep + mesh_file_npy)
        foe = np.load(path_mesh + os.sep + mesh_file_npy, allow_pickle=True)
        mesh = foe.item()
        #print(mesh['mesh_0']['bari'].keys())
        #sys.exit()
    else:
        print('Load slab meshes from .dat files:                       <------', path_mesh + os.sep + '*.dat' )



        #Load mesh. Compute reference utm (reference point is the first one of mesh_name)
        mesh_0 = load_mesh(mesh_face=faces[0], mesh_node=nodes[0], mesh_name=names[0], path=path_mesh)
        mesh_1 = load_mesh(mesh_face=faces[1], mesh_node=nodes[1], mesh_name=names[1], path=path_mesh)
        mesh_2 = load_mesh(mesh_face=faces[2], mesh_node=nodes[2], mesh_name=names[2], path=path_mesh)


        # Compute utm coordinates of the berycenter of each face
        mesh_0 = set_baricenter(mesh=mesh_0)
        mesh_1 = set_baricenter(mesh=mesh_1)
        mesh_2 = set_baricenter(mesh=mesh_2)

        mesh = {'mesh_0' : mesh_0, 'mesh_1' : mesh_1, 'mesh_2' : mesh_2}

#        print('Created dictionary for slab meshes                      ------> ', path_mesh + os.sep + mesh_file_npy + '.npy')
#             #save dictionary in npy file
#        np.save(path_mesh + os.sep + mesh_file_npy, mesh)


    ############################################
    #####Load Regionalization#####
    if os.path.isfile(regionalization_npy) and args.preload != 'Yes':
        print('Load dictionary for regionalization:                    <------ ', regionalization_npy)
        # load npy file into a dictionary. The option allow_pickle=True is needed if the npy file contains np.array
        foe = np.load(regionalization_npy, allow_pickle=True)
        regionalization = foe.item()
        print("%64s Regions found: %3d" % ('', regionalization['Npoly']))

    else:

        print(args.preload)
        print(regionalization_npy)

        print('Loading regionalization from SPTHA project {}'.format(project_name), "  <------ ", file_regions)
        f = open(file_regions, "r")

        regionalization = {} #=dict()
        ind = []
        ID = []
        Tnames = []
        Ttypes = []
        Tleng = []
        Tpoint = []
        Tlon = []
        Tlat = []

        number = 0

        ### build dictionary
        for lines in f:
            foe = lines.rstrip().split(':')

            number += 1
            if(args.verbose == True):
                print('         Reading information for region {} ({})'.format(("%3d" % number),foe[1]))

            nvertex,coords,lon,lat = region_coordinates_points_splitter_2_dictionary(points=foe[3])
            seismicity_type = region_type_number_2_array(numbers=foe[2])

            ind.append(number)
            ID.append(foe[0])
            Tnames.append(foe[1])
            Ttypes.append(seismicity_type)
            Tleng.append(nvertex)
            Tpoint.append(coords)
            Tlon.append(lon)
            Tlat.append(lat)

            #region = {number: {'label':foe[0], 'region_name': foe[1], 'seismicity_type': seismicity_type, 'coordinates' : region_coordinates},\
            #     'nr' : number}
            tmpreg = {'ind': ind,'ID': ID,'Tnames': Tnames, 'Ttypes': Ttypes, 'Tleng': Tleng, 'Tlon': Tlon, 'Tlat': Tlat, 'Tpoint': Tpoint, 'Npoly': number}
            regionalization.update(tmpreg)

        f.close()

#        np.save(regionalization_npy, regionalization)
#        print('Created dictionary for regionalization                  ------> ', regionalization_npy)

    #####End Regionalization#####
    ############################################


    ############################################
    #####Load Points of Interest#####
    if os.path.isfile(pois_npy) and args.preload != 'Yes':
        print('Load dictionary with POIs list:                         <------ ', pois_npy)
        # load npy file into a dictionary. The option allow_pickle=True is needed if the npy file contains np.array
        foe = np.load(pois_npy, allow_pickle=True)
        POIs = foe.item()
        empty_space = "%64s" % ('')
        print(empty_space, '{} POIs found'.format(len(POIs['name'])))
        print(empty_space, '--> {} in the Mediterranean Sea'.format(countX(POIs['Mediterranean'], 1)))
        print(empty_space, '--> {} in the Black Sea'.format(countX(POIs['BlackSea'], 1)))
        print(empty_space, '--> {} in the Atlantic Ocean'.format(countX(POIs['Atlantic'], 1)))


    else:

        print('Loading POIs list from SPTHA project {}'.format(project_name), '        <------ ', file_pois)

        f = open(file_pois, 'r')

        POIs = {} #=dict()
        name = []
        lon = []
        lat = []
        dep = []
        Mediterranean = []
        BlackSea = []
        Atlantic = []

        ### build dictionary
        for lines in f:
            foe = lines.rstrip().split(' ')
            #print(foe[0])

            name.append(foe[0])
            lon.append(float(foe[1]))
            lat.append(float(foe[2]))
            dep.append(float(foe[3]))

            if 'blk' in foe[0]:
                Mediterranean.append(0)
                BlackSea.append(1)
                Atlantic.append(0)
            elif 'med' in foe[0]:
                Mediterranean.append(1)
                BlackSea.append(0)
                Atlantic.append(0)
            elif 'nea' in foe[0]:
                Mediterranean.append(0)
                BlackSea.append(0)
                Atlantic.append(1)
            else:
                print("No POIS definition found in ", file_pois)
                sys.exit()

            tmppois = {'name': name, 'lon': lon, 'lat': lat, 'dep': dep, 'Mediterranean': Mediterranean, 'BlackSea': BlackSea, 'Atlantic': Atlantic}
            POIs.update(tmppois)

        print(empty_space, '{} POIs found'.format(len(POIs['name'])))
        print(empty_space, '--> {} in the Mediterranean Sea'.format(countX(POIs['Mediterranean'], 1)))
        print(empty_space, '--> {} in the Black Sea'.format(countX(POIs['BlackSea'], 1)))
        print(empty_space, '--> {} in the Atlantic Ocean'.format(countX(POIs['Atlantic'], 1)))
#        poi_dictionary_file = os.path.join(INPDIR,'POIs')

#        print('Created dictionary for POIs list                        ------> ', pois_npy + '.npy')
             #save dictionary in npy file
#        np.save(pois_npy, POIs)

        f.close()
    #####End POIs#####
    ############################################

    ############################################
    #####Load Discretization#####
    if os.path.isfile(discretization_npy) and args.preload != 'Yes' and args.preload_discretization != 'Yes':
        print('Loading discretization dictionary                       <------ ', discretization_npy)
        foe = np.load(discretization_npy, allow_pickle=True)
        discretizations = foe.item()
        for i in discretizations.keys():
            print(empty_space, i)




    else:

        print('Loading discretizations from SPTHA project {}'.format(project_name), '  <------', discretization_dir)

        # Initialize dicretization dictionary
        discretizations = {} #=dict()

        # Load into array the poligon of the T points
        list_poly = []
        npoly=regionalization['Npoly']
        for j in range(npoly):
            poly= geometry.Polygon(regionalization['Tpoint'][j])
            list_poly.append(poly)
        multi_poly = geometry.MultiPolygon(list_poly)

        print("Load Moho %-45s <------ %s" % ('',file_moho))
        moho_grid_array, moho_all, moho_dep = load_moho_grid_array_and_depth(infile=file_moho, info='moho', lon_pos=1, lat_pos=2)

        barycenter_grid_array, barycenter_depth, barycenter_position = load_barycenter_file(cfg=Config)

        for inode in range(len(bs_nodes)):

            filename = bs_nodes[inode] + '.txt'
            if (inode == 2): filename = bs_nodes[inode] + '_full.txt'
            file_discr =  os.path.join(discretization_dir, filename)
            print("Load %-50s <------ %s" % (bs_nodes[inode],file_discr))
            discr_grid_array = load_grid_array(infile = file_discr, info='discr', lon_pos=1, lat_pos=2, inode=inode)


            items = get_BS_discretizations(i            = inode, \
                                          regions_poly = multi_poly, \
                                          IDreg        = regionalization['ID'], \
                                          mz           = moho_dep, \
                                          tmpdiscr     = discretizations, \
                                          file_name    = file_discr, \
                                          grid_moho    = moho_grid_array, \
                                          grid_discr   = discr_grid_array, \
                                          moho_all     = moho_all)
                                          #mxy          = moho_pos,  useless now
            tmpdiscr = {bs_nodes[inode]: items}
            discretizations.update(tmpdiscr)


        for inode in range(len(ps_nodes)):
            filename = ps_nodes[inode] + '.txt'
            file_discr =  os.path.join(discretization_dir, filename)
            print("Load %-50s <------ %s" % (ps_nodes[inode],file_discr))
            items = get_PS_discretizations(i=inode,file_name=file_discr,regions_poly=multi_poly,bz=barycenter_depth,grid_bary=barycenter_grid_array, bxy=barycenter_position)
            tmpdiscr = {ps_nodes[inode]: items}
            discretizations.update(tmpdiscr)

           #f.close()

        #FIND ACTIVE REGIONS FOR SEISMICITY TYPE
        bsyn = np.zeros(npoly)
        psyn = np.zeros(npoly)
        bsyn[np.unique(discretizations['BS-2_Position']['Region'])-1] = 1
        psyn[np.unique(discretizations['PS-2_PositionArea']['Region'])-1] = 1

        tmpyn = {'BSexistYN': bsyn, 'PSexistYN': psyn}
        discretizations.update(tmpyn)

#        print('Created Dictionary discretization                       ------>', discretization_npy)
#        np.save(discretization_npy, discretizations)
    
    #####End Discretization#####
    ############################################



    ############################################
    #####Load Model Weights#####
    if os.path.isfile(weight_npy) and args.preload != 'Yes' and args.preload_weight != 'Yes':
        print('Loading model weights dictionary                        <------ ', weight_npy)
        foe = np.load(weight_npy, allow_pickle=True)
        modelweights = foe.item()
        for k in modelweights.keys():
                print(empty_space, k)

        # select only one type of ps_weigth
        # Only one weigth mode can be selected
        #if(int(args.ps_type) != -1):
        selected_index = np.where(modelweights['BS1_Mag']['Type'] == int(args.ps_type))
        modelweights['BS1_Mag']['Type'] = modelweights['BS1_Mag']['Type'][selected_index]
        modelweights['BS1_Mag']['Wei'] = modelweights['BS1_Mag']['Wei'][selected_index]

        selected_index = np.where(modelweights['PS2_Bar']['Type'] == int(args.ps_type))
        modelweights['PS2_Bar']['Type'] = modelweights['PS2_Bar']['Type'][selected_index]
        modelweights['PS2_Bar']['Wei'] = modelweights['PS2_Bar']['Wei'][selected_index]

    else:

        print('Loading model weights from SPTHA project {}'.format(project_name), '    <------', weight_mat)

        s   = dict()
        D   = dict()
        modelweights = dict()

        matf = weight_mat

        #Initialize dict
        #modelweights                    = dict()
        modelweights['BS1_Mag']         = dict()
        modelweights['BS2_Pos']         = dict()
        modelweights['BS3_Depth']       = dict()
        modelweights['BS4_FocMech']     = dict()
        modelweights['BS5_AreaLength']  = dict()
        modelweights['BS6_Slip']        = dict()
        modelweights['PS1_Mag']         = dict()
        modelweights['PS2_Bar']         = dict()
        ###
        f = loadmat(matf)['ModelsWeight']['BS1_Mag']['Wei']
        s['Wei'] = np.squeeze(f)
        f = loadmat(matf)['ModelsWeight']['BS1_Mag']['Type']
        s['Type'] = np.squeeze(f)
        modelweights['BS1_Mag']['Wei']  = s['Wei']
        modelweights['BS1_Mag']['Type'] = s['Type']

        ###
        f = loadmat(matf)['ModelsWeight']['BS2_Pos']['Wei']
        s['Wei'] = np.squeeze(f)
        f = loadmat(matf)['ModelsWeight']['BS2_Pos']['Type']
        s['Type'] = np.squeeze(f)
        modelweights['BS2_Pos']['Wei']  = s['Wei']
        modelweights['BS2_Pos']['Type'] = s['Type']

        ###
        f = loadmat(matf)['ModelsWeight']['BS3_Depth']['Wei']
        s['Wei'] = np.squeeze(f)
        modelweights['BS3_Depth']['Wei']  = s['Wei']

        ###
        f = loadmat(matf)['ModelsWeight']['BS4_FocMech']['Wei']
        s['Wei'] = np.squeeze(f)
        f = loadmat(matf)['ModelsWeight']['BS4_FocMech']['Type']
        s['Type'] = np.squeeze(f)
        modelweights['BS4_FocMech']['Wei']  = s['Wei']
        modelweights['BS4_FocMech']['Type']  = s['Type']

        ###
        f = loadmat(matf)['ModelsWeight']['BS5_AreaLength']['Wei']
        s['Wei'] = np.squeeze(f)
        modelweights['BS5_AreaLength']['Wei']  = s['Wei']

        ###
        f = loadmat(matf)['ModelsWeight']['BS6_Slip']['Wei']
        s['Wei'] = np.squeeze(f)
        modelweights['BS6_Slip']['Wei']  = s['Wei']

        ###
        f = loadmat(matf)['ModelsWeight']['PS1_Mag']['Wei']
        s['Wei'] = np.squeeze(f)
        f = loadmat(matf)['ModelsWeight']['PS1_Mag']['Type']
        s['Type'] = np.squeeze(f)
        modelweights['PS1_Mag']['Wei']  = s['Wei']
        modelweights['PS1_Mag']['Type'] = s['Type']


        ###
        f = loadmat(matf)['ModelsWeight']['PS2_Bar']['Wei']
        s['Wei'] = np.squeeze(f)
        f = loadmat(matf)['ModelsWeight']['PS2_Bar']['Type']
        s['Type'] = np.squeeze(f)
        #modelweights['PS2_Bar']['Wei']  = s['Wei']
        #modelweights['PS2_Bar']['Type'] = s['Type']
        modelweights['PS2_Bar']['Wei']  = s['Wei']
        modelweights['PS2_Bar']['Type'] = s['Type']

#        print('Created dictionary model weights                        ------>', weight_npy)
#        np.save(weight_npy, modelweights)

    #####End Model Weights#####
    ############################################

    ############################################
    #####Load Lookup Table#####
    """
    if os.path.isfile(lookup_tables_npy) and args.preload != 'Yes':
        print('Loading lookup table for conditional hazard dictionary  <------ ', lookup_tables_npy)
        # load npy file into a dictionary. The option allow_pickle=True is needed if the npy file contains np.array
        foe = np.load(lookup_tables_npy, allow_pickle=True)
        LookupTableConditionalHazardCurves = foe.item()
        for i in LookupTableConditionalHazardCurves.keys():
            print(empty_space, i)


    else:
        print('Loading lookup table for conditional hazard curves from SPTHA project {}'.format(project_name))

        # 3 files to load
        LookupTableConditionalHazardCurves = {} #=dict()

        print("Load %-50s <------ %s" % ('', file_mihthr))
        tmptable = load_lookup_tables_files(file=file_mihthr, tag='HazardCurveThresholds')
        LookupTableConditionalHazardCurves.update(tmptable)

        print("Load %-50s <------ %s" % ('', file_mihthr))
        tmptable = load_lookup_tables_files(file=file_mihsteps, tag='InputIntensities')
        LookupTableConditionalHazardCurves.update(tmptable)

        print("Load %-50s <------ %s" % ('', file_mihthr))
        tmptable = load_lookup_tables_files(file=table_file, tag='LookupTable')
        LookupTableConditionalHazardCurves.update(tmptable)

        LookupTableConditionalHazardCurves['CondHazMean'] = np.squeeze(np.mean(LookupTableConditionalHazardCurves['LookupTable'], axis=2))  ##CHECK SHAPE!!!!###

        print('Created dictionary for HC lookup table                  ------>', lookup_tables_npy)
        np.save(lookup_tables_npy, LookupTableConditionalHazardCurves)


    thrs = LookupTableConditionalHazardCurves["HazardCurveThresholds"]
    lut = np.zeros((len(thrs),len(thrs)))
    for i in range(len(thrs)):
        lut[:i,i] = 1
    """
    #####End Lookup Table#####
    ############################################

     ############################################
    #STORING INFORMATION FROM SPTHA IN A DICTIONARY
    ############################################
    # LongTermInfo: Contains all the information coming from longterm SPTHA
    LongTermInfo                                          = {}
    LongTermInfo['ProjectName']                           = project_name
    LongTermInfo['Regionalization']                       = regionalization
    #LongTermInfo['POIs']                                  = POIs
    LongTermInfo['Discretizations']                       = discretizations
    LongTermInfo['Model_Weights']                         = modelweights
    #LongTermInfo['LookupTableConditionalHazardCurves']    = LookupTableConditionalHazardCurves
    #LongTermInfo['GenericConditionalHazardCurves']        = lut
    # fb domanda che sarebbe? perch? indici in quel modo?
    LongTermInfo['vecID']                                 = 10.**np.array([8, 5, 2, 0, -2, -4, -6])   # vector for quick search on parameters ?????????



    ############################################
    #PRE-PROCESSING BS-4 FOC MECH PROBABILITIES,
    #TO OBTAIN AVERAGE PROB IN
    ############################################
    # MeanProb_BS4_FocMech_Reg

    # set region file_mihstep
    region_to_ignore = Config.get('mix_parameters','ignore_regions').split()
    region_ps_1      = ast.literal_eval(Config.get('regionsPerPS','1'))
    region_ps_2      = ast.literal_eval(Config.get('regionsPerPS','2'))
    region_ps_3      = ast.literal_eval(Config.get('regionsPerPS','3'))
    region_to_ignore = list(map(int, region_to_ignore))
    region_list      = [x for x in range(regionalization['Npoly']) if x not in region_to_ignore]
    region_listPs    = [-1 for x in range(regionalization['Npoly']) if x not in region_to_ignore]


    for i in region_ps_1:
        region_listPs[i-1] = 1
    for i in region_ps_2:
        region_listPs[i-1] = 2    #print(i, region_listPs[i])
    for i in region_ps_3:
        region_listPs[i-1] = 3    #region_listPs[region_ps_1[i]-1] = 1

    LongTermInfo['region_listPs'] = region_listPs

    # Set dir for focal mechanism and PROBABILITIES
    #mech_dir = Config.get('Folders','Folder_Preproc')
    pyptf_focal_mechanism_dir = Config.get('pyptf','FocMech_Preproc')
    if not os.path.exists(pyptf_focal_mechanism_dir):
        os.makedirs(pyptf_focal_mechanism_dir)
    #prob_dir = Config.get('Folders','Folder_ProbMod')
#    pyptf_probability_models_dir = Config.get('pyptf','ProbabilityModels')
#    if not os.path.exists(pyptf_probability_models_dir):
#        os.makedirs(pyptf_probability_models_dir)
    tsumaps_probability_models_dir = Config.get('tsumaps','ProbabilityModels')


    focal_mechanism_root_name    = Config.get('Files','focal_mechanism_root_name')
    probability_models_root_name = Config.get('Files','probability_models_root_name')


    # Find regions with and withouth BS-4
    regions_without_bs_focal_mechanism = []
    regions_with_bs_focal_mechanism    = []
    for iReg in range(len(region_list)):
        if (sum(itype for itype in regionalization['Ttypes'][iReg] if itype == 1) > 0):
            regions_with_bs_focal_mechanism.append(iReg)
        else:
            regions_without_bs_focal_mechanism.append(iReg)

    ModelsProb_Region_files = dict()

    print("Loading MeanProb_BS4_FocMech dictionaries               <------ ", focal_mechanism_root_name, "and",  probability_models_root_name)

    for iReg in regions_with_bs_focal_mechanism:

        # define files
        filename  = focal_mechanism_root_name + '{}'.format(str(iReg+1).zfill(3)) + '.npy'
        f_FocMech = os.path.join(pyptf_focal_mechanism_dir,filename)
        filename  = probability_models_root_name + '{}'.format(str(iReg+1).zfill(3)) + '_' +  regionalization['ID'][iReg] + '.mat'
        f_ProbMod = os.path.join(tsumaps_probability_models_dir,filename)
        #print(f_FocMech)
        #print(f_ProbMod)
        #sys.exit()

        #print("Load                                                    <------ ", f_ProbMod)
        ModelsProb_Region_files = ModelsProb_Region_hdf5_to_py(hdf5=f_ProbMod, npy=f_FocMech, \
                                                               dictionary=ModelsProb_Region_files, \
                                                               modelweights=modelweights, \
                                                               discretizations=discretizations, \
                                                               args=args)

        #print(ModelsProb_Region_files)
        #sys.exit()

    # This is just to have a simple way to select files in probability_scenarios
    for iReg in regions_without_bs_focal_mechanism:

        # define files
        filename  = focal_mechanism_root_name + '{}'.format(str(iReg+1).zfill(3)) + '.npy'
        f_FocMech = os.path.join(pyptf_focal_mechanism_dir,filename)

        empty = dict()
        if(os.path.isfile(f_FocMech)):
            pass
        else:
            np.save(f_FocMech, empty, allow_pickle=True)
            print("Created empty", f_FocMech)

        ModelsProb_Region_files['ModelsProb_Region_files'].append(f_FocMech)

    ModelsProb_Region_files['ModelsProb_Region_files'].sort()
    ModelsProb_Region_files['regions_with_bs_focal_mechanism']    = regions_with_bs_focal_mechanism
    ModelsProb_Region_files['regions_without_bs_focal_mechanism'] = regions_without_bs_focal_mechanism




    ############################################
    #PSBarInfo
    ############################################
    #PSBarInfo_mat = Config.get('tsumaps','PSBarInfo')


    ## If PSBarInfo_py exists and preload == no pass
    #if (os.path.isfile(PSBarInfo_py) == False and args.preload == 'Yes' or args.preload_PS == 'Yes'):
    #    print("Loading MeanProb_BS4_FocMech dictionaries               <------ ", focal_mechanism_root_name, "and",  probability_models_root_name)
    #    print("Converting ", PSBarInfo_mat, "                          <------ ", PSBarInfo_py)
    #    PSBarInfo_Dict = PSBarInfo_mat2py(matfile=PSBarInfo_mat, pyfile=PSBarInfo_py, args=args, cfg=Config)
    #    np.save(PSBarInfo_py, PSBarInfo_Dict, allow_pickle=True)
    #else:
    #    PSBarInfo_Dict = np.load(PSBarInfo_py, allow_pickle=True).item()
    #    #print(PSBarInfo_Dict['BarPSperModelReg'][15][0])
    #    #print("--")
    #    #print(PSBarInfo_Dict['BarPSperModelReg'][15][1])
    #    #print("--")
    #    #print(PSBarInfo_Dict['BarPSperModelReg'][15][2])

    #    #sys.exit()

    POIs = select_pois_and_regions(cfg=Config, args=args, pois_dictionary=POIs, regionalization_dictionary=regionalization)


    #return LongTermInfo, POIs, PSBarInfo_Dict, mesh, ModelsProb_Region_files
    return LongTermInfo, POIs, mesh, ModelsProb_Region_files
