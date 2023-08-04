import sys
import utm
import scipy
import ast
import numpy as np

from itertools import chain

#from numba import jit

#@jit(nopython=True)
def find_distances_tetra_mesh(**kwargs):

    mesh   = kwargs.get('mesh', None)
    tetra  = kwargs.get('tetra', None)
    buffer = kwargs.get('buffer', None)
    moho   = kwargs.get('moho', None)
    g_moho = kwargs.get('grid_moho', None)

    d = dict()

    dist   = np.zeros(len(tetra))
    m_dist = np.zeros(len(tetra))

    #print(np.shape(mesh))
    #print(np.shape(tetra[0]))
    #sys.exit()
    for i in range(len(tetra)):
        dist[i]   = np.amin(np.linalg.norm(mesh - tetra[i], axis=1))
        m_dist[i] = np.amin(np.linalg.norm(g_moho - tetra[i], axis=1))

    # Check if all below moho
    d['tetra_in_moho']        = True #default

    # Minimal distance
    d['distances_mesh_tetra'] = dist
    d['distance_min_value']   = np.amin(dist)
    d['distance_min_idx']     = np.argmin(dist)
    d['moho_d_mesh_tetra']    = m_dist
    d['moho_d_min_value']     = np.amin(m_dist)
    d['moho_d_min_idx']       = np.argmin(m_dist)


    # All distances min than buffer
    d['idx_less_then_buffer'] = np.where(dist <= buffer)[0]
    d['idx_more_then_buffer'] = np.where(dist >  buffer)[0]

    ## Questa parte qui general 'errore'!!!!
    # select all indx below the surface for the ones into the slab (PS)
    tmp_tetra = np.take(tetra[:,2],d['idx_less_then_buffer'])
    d['tmp']  = np.where(tmp_tetra <= 0)[0]
    d['idx_less_then_buffer_effective'] = d['idx_less_then_buffer'][d['tmp']]

    # select all indx below the surface for the ones outside the slab (BS)
    tmp_tetra = np.take(tetra[:,2],d['idx_more_then_buffer'])
    d['tmp']  = np.where(tmp_tetra <= 0)[0]
    d['idx_more_then_buffer_effective'] = d['idx_more_then_buffer'][d['tmp']]

    tmp_tetra = np.take(tetra[:,2],d['idx_more_then_buffer_effective'])
    tmp_moho  = np.take(moho, d['idx_more_then_buffer_effective'])
    d['tmp']  = np.where((tmp_moho -1*tmp_tetra/1000) <=0)
    d['idx_more_then_buffer_effective'] = d['idx_more_then_buffer_effective'][d['tmp']]

    # Check if this is in moho
    if(len(d['idx_more_then_buffer_effective']) == 0):
        d['tetra_in_moho'] = False

    return d

def find_tetra_index_for_ps_and_bs(**kwargs):

    Config         = kwargs.get('cfg', None)
    ee             = kwargs.get('event_parameters', None)
    lambda_bsps    = kwargs.get('lambda_bsps', None)
    mesh           = kwargs.get('mesh', None)
    moho           = kwargs.get('moho', None)
    grid_moho      = kwargs.get('grid_moho', None)

    buffer         = float(Config.get('lambda','subd_buffer'))


    tt = np.empty((0,3))
    min_distance = []
    slabs        = []

    #print(grid_moho[:,0])
    #print(len(grid_moho))
    # here make 1 grid moho in utm
    grid_moho_utm = utm.from_latlon(grid_moho[:,1], grid_moho[:,0], ee['ee_utm'][2])
    grid_moho     = np.column_stack((grid_moho_utm[0].transpose(), grid_moho_utm[1].transpose(), (1000*grid_moho[:,2]).transpose()))

    for keys in mesh:

        # Convert lat lon to utm for the baricenter
        mesh[keys]['bari']['utm'] = utm.from_latlon(mesh[keys]['bari']['lat'], mesh[keys]['bari']['lon'], ee['ee_utm'][2])

        tmp_mesh = np.column_stack((mesh[keys]['bari']['utm'][0].transpose(), \
                                    mesh[keys]['bari']['utm'][1].transpose(), \
                                    mesh[keys]['bari']['depth'].transpose()))
        tt_mesh  = np.concatenate((tt, tmp_mesh))

        mesh[keys]['d_dist'] = find_distances_tetra_mesh(mesh = tt_mesh, tetra = lambda_bsps['tetra_xyz'], buffer = buffer, moho = moho, grid_moho = grid_moho)
        print('     --> Min distance from slab %s %10.3f [km]' % (mesh[keys]['name'], mesh[keys]['d_dist']['distance_min_value']/1000))
        print('         --> Nr of PS tetra with dist.  < %4.1f [km] from slab %s : %d  (effective: %d)' % \
              (buffer/1000, mesh[keys]['name'], len(mesh[keys]['d_dist']['idx_less_then_buffer']), len(mesh[keys]['d_dist']['idx_less_then_buffer_effective'])))
        print('         --> Nr of BS tetra with dist. >= %4.1f [km] from slab %s : %d  (effective: %d)' % \
              (buffer/1000, mesh[keys]['name'], len(mesh[keys]['d_dist']['idx_more_then_buffer']), len(mesh[keys]['d_dist']['idx_more_then_buffer_effective'])))

    return mesh

def compute_ps_bs_gaussians_general(**kwargs):

    Config         = kwargs.get('cfg', None)
    vol            = kwargs.get('vol', None)
    ee             = kwargs.get('event_parameters', None)
    mesh           = kwargs.get('mesh', None)
    tetra          = kwargs.get('tetra', None)
    lambda_bsps    = kwargs.get('lambda_bsps', None)

    buffer         = float(Config.get('lambda','subd_buffer'))

    hx             = ee['ee_utm'][0]
    hy             = ee['ee_utm'][1]
    hz             = ee['depth']* (1000.0)
    covariance     = ee['PosCovMat_3dm']
    xyz            = np.array([hx, hy, hz])

    #First concatenate ps arrays for all meshes
    ps_first = np.zeros(3)
    bs_first = np.zeros(3)
    pb_first = np.zeros(3)

    # first merge index
    ps_idx    = []
    bs_idx    = []
    bs_ps_idx = []
    gauss_ps_eff = np.array([])
    gauss_bs_eff = np.array([])

    #distances min
    min_d_mesh = sys.float_info.max
    min_d_moho = sys.float_info.max

    # Some booleans
    inmoho    = False  # True if tetra in moho
    lmbda_mix = False  # True if lambda = ]1,0[

    for keys in mesh:
        ps_idx.extend((mesh[keys]['d_dist']['idx_less_then_buffer_effective']).tolist())
        bs_idx.extend((mesh[keys]['d_dist']['idx_more_then_buffer_effective']).tolist())
        if(mesh[keys]['d_dist']['tetra_in_moho'] == True):
            inmoho = True
        if(mesh[keys]['d_dist']['moho_d_min_value'] <= min_d_moho):
            min_d_moho = mesh[keys]['d_dist']['moho_d_min_value']
        if(mesh[keys]['d_dist']['distance_min_value'] <= min_d_mesh):
            min_d_mesh = mesh[keys]['d_dist']['distance_min_value']
        #print(mesh[keys]['d_dist']['moho_d_min_value'])

    #print(min_d_moho, min_d_mesh)
    #sys.exit()
    # First check:
    # 1. Se tetra nella moho ma non a contatto con nessuna mash: BS=1, PS=0
    # 2. Se tetra sotto la moho e piu vicino alla moho che a mesh: BS=1, PS=0
    # 3. Se tetra sotto la moho e piu distante alla moho che a mesh: BS=0, PS=1
    # 4. Se nella moho e ci sono indici idx: calcola differenze
    # If tetra in moho (d['tetra_in_moho'] = True) and len(ps_idx) == 0 ==> All bs
    lmbda_mix = False
    if(inmoho == True and len(ps_idx) == 0):
        lambda_ps = 0.0
        lambda_bs = 1.0
    elif(inmoho == False and min_d_moho-buffer <= min_d_mesh):
        lambda_ps = 0.0
        lambda_bs = 1.0
    elif(inmoho == False and min_d_moho-buffer > min_d_mesh):
        lambda_ps = 1.0
        lambda_bs = 0.0
    else:
        lmbda_mix = True
        ps_idx    = set(ps_idx)
        bs_idx    = set(bs_idx) - set(ps_idx)
        #bs_idx    = set(bs_idx)
        bs_ps_idx = set(chain(ps_idx,bs_idx))


        ps_idx    = np.array(list(ps_idx))
        bs_idx    = np.array(list(bs_idx))
        bs_ps_idx = np.array(list(bs_ps_idx))

        ps_tetra    = tetra[ps_idx]
        bs_tetra    = tetra[bs_idx]
        bs_ps_tetra = tetra[bs_ps_idx]

        ps_tetra[:,2]    = ps_tetra[:,2]* -1
        bs_tetra[:,2]    = bs_tetra[:,2]* -1
        bs_ps_tetra[:,2] = bs_ps_tetra[:,2]* -1

        gauss_ps_eff     = scipy.stats.multivariate_normal.pdf(ps_tetra, xyz, covariance)
        gauss_bs_eff     = scipy.stats.multivariate_normal.pdf(bs_tetra, xyz, covariance)
        gauss_bs_ps_eff  = scipy.stats.multivariate_normal.pdf(bs_ps_tetra, xyz, covariance)

        sum_bs_ps = np.sum(np.multiply(gauss_bs_ps_eff,vol[bs_ps_idx]))
        sum_ps    = np.sum(np.multiply(gauss_ps_eff,vol[ps_idx]))
        sum_bs    = np.sum(np.multiply(gauss_bs_eff,vol[bs_idx]))

        lambda_ps = np.sum(np.multiply(gauss_ps_eff,vol[ps_idx])) / sum_bs_ps
        lambda_bs = np.sum(np.multiply(gauss_bs_eff,vol[bs_idx])) / sum_bs_ps


    #print(lambda_ps)
    #sys.exit()
    lambda_bsps['lambda_ps']  = lambda_ps
    lambda_bsps['lambda_bs']  = lambda_bs
    lambda_bsps['gauss_ps']   = gauss_ps_eff
    lambda_bsps['gauss_bs']   = gauss_bs_eff
    lambda_bsps['lmbda_mix']  = lmbda_mix


    print(" --> lambda PS: %6.4e      Volume ps:    %10.4e [m^3]" % (lambda_ps, np.sum(vol[ps_idx])))
    print(" --> lambda BS: %6.4e      Volume bs:    %10.4e [m^3]" % (lambda_bs, np.sum(vol[bs_idx])))
    print(" -->            %8s        Volume bs-ps: %10.4e [m^3]" % (' ',np.sum(vol[bs_ps_idx])))

    return lambda_bsps

def compute_ps_bs_gaussians_single_zone(**kwargs):

    Config         = kwargs.get('cfg', None)
    vol            = kwargs.get('vol', None)
    ee             = kwargs.get('event_parameters', None)
    mesh           = kwargs.get('mesh', None)
    tetra          = kwargs.get('tetra', None)
    lambda_bsps    = kwargs.get('lambda_bsps', None)

    hx             = ee['ee_utm'][0]
    hy             = ee['ee_utm'][1]
    hz             = ee['depth']* (1000.0)
    covariance     = ee['PosCovMat_3dm']
    xyz            = np.array([hx, hy, hz])
    lambda_ps_sub  = []

    #print('............................',lambda_bsps['lmbda_mix'])
    #sys.exit()

    if(lambda_bsps['lmbda_mix'] == False):

        lambda_bsps['lambda_ps_sub']       = [0,0,0]
        lambda_bsps['lambda_ps_on_ps_tot'] = [0,0,0] # Fixed for lambda_mix == False (PS == 0)

        return lambda_bsps

    #print(mesh['mesh_0']['name'])
    #sys.exit()


    for keys in mesh:

        ps_first = np.zeros(3)
        bs_first = np.zeros(3)
        pb_first = np.zeros(3)

        # first merge index
        ps_idx    = []
        bs_idx    = []
        bs_ps_idx = []

        # first Compute general PS-BS
        #for keys in mesh:
        ps_idx.extend((mesh[keys]['d_dist']['idx_less_then_buffer_effective']).tolist())
        bs_idx.extend((mesh[keys]['d_dist']['idx_more_then_buffer_effective']).tolist())
        if(len(ps_idx) == 0):
            lambda_ps = 0.0
            lambda_ps_sub.append(lambda_ps)
            print("     --> Single %-5s lambda PS: %6.4e      Volume ps:    %10.4e [m^3]" % (mesh[keys]['name'], lambda_ps, np.sum(vol[ps_idx])))

        else:
            ps_idx    = set(ps_idx)

            ps_idx    = np.array(list(ps_idx))

            ps_tetra    = tetra[ps_idx]

            ps_tetra[:,2]    = ps_tetra[:,2]* -1

            gauss_ps_eff     = scipy.stats.multivariate_normal.pdf(ps_tetra, xyz, covariance)

            sum_ps    = np.sum(np.multiply(gauss_ps_eff,vol[ps_idx]))
            lambda_ps = (np.sum(np.multiply(gauss_ps_eff,vol[ps_idx])) / sum_ps) * lambda_bsps['lambda_ps']
            lambda_ps_sub.append(lambda_ps)

            print("     --> Single %-5s lambda PS: %6.4e      Volume ps:    %10.4e [m^3]" % (mesh[keys]['name'], lambda_ps, np.sum(vol[ps_idx])))



    lambda_bsps['lambda_ps_sub'] = lambda_ps_sub
    #print(">>>>>>>>>>>>>>>>>> ciaooo", lambda_bsps['lambda_ps_on_ps_tot'])

    # Define LambdsPs for each reagion on total lambda PS
    lambda_bsps['lambda_ps_on_ps_tot'] =  lambda_bsps['lambda_ps_sub'] / lambda_bsps['lambda_ps']


    return lambda_bsps

def update_lambda_bsps_dict(**kwargs):

    Config          = kwargs.get('cfg', None)
    lambda_bsps     = kwargs.get('lambda_bsps', None)
    Regionalization = kwargs.get('Regionalization', None)

    mesh_zones      = ast.literal_eval(Config.get('lambda','mesh_zones'))

    """
    SettingsLambdaBSPS.regionsPerPS = nan(LongTermInfo.Regionalization.Npoly,1);
    SettingsLambdaBSPS.regionsPerPS([3,24,44,48,49])=1;
    SettingsLambdaBSPS.regionsPerPS([10,16,54])=2;
    SettingsLambdaBSPS.regionsPerPS([27,33,35,36])=3;

    config['lambda']['mesh_zones']   = '{\'0\':\'[3,24,44,48,49]\', \'1\':\'[10,16,54]\', \'2\':\'[27,33,35,36]\'}'
    """
    #print(Regionalization['Npoly'])
    regionsPerPS    = np.empty(Regionalization['Npoly'])
    regionsPerPS[:] = np.NaN


    for key in mesh_zones:

        l = ast.literal_eval(mesh_zones[key])
        regionsPerPS[l] = int(key)
        #print(type(ast.literal_eval(mesh_zones[key])))


    lambda_bsps['regionsPerPS'] = regionsPerPS


    return lambda_bsps

def separation_lambda_BSPS(**kwargs):

    Config          = kwargs.get('cfg', None)
    ee              = kwargs.get('event_parameters', None)
    args            = kwargs.get('args', None)
    LongTerm        = kwargs.get('LongTermInfo', None)
    lambda_bsps     = kwargs.get('lambda_bsps', None)
    #Regionalization = kwargs.get('Regionalization', None)
    mesh            = kwargs.get('mesh', None)



    moho_ll  = np.column_stack((LongTerm['Discretizations']['BS-2_Position']['Val_x'], LongTerm['Discretizations']['BS-2_Position']['Val_y']))
    tetra_ll = np.column_stack((lambda_bsps['tetra_bar']['lon'], lambda_bsps['tetra_bar']['lat']))


    bar_depth_moho = scipy.interpolate.griddata(moho_ll,
                                                LongTerm['Discretizations']['BS-2_Position']['DepthMoho'],
                                                tetra_ll)


    print(' --> Distance between tetra and slabs:')
    mesh = find_tetra_index_for_ps_and_bs(event_parameters = ee, \
                                          lambda_bsps      = lambda_bsps, \
                                          mesh             = mesh, \
                                          cfg              = Config, \
                                          moho             = bar_depth_moho, \
                                          grid_moho        = LongTerm['Discretizations']['BS-2_Position']['grid_moho'])
    #print(mesh['mesh_0']['name'], mesh['mesh_0']['nodes']['lat'][:2], mesh['mesh_0']['nodes']['lon'][:2])
    #print(mesh['mesh_1']['name'], mesh['mesh_1']['nodes']['lat'][:2], mesh['mesh_1']['nodes']['lon'][:2])
    #print(mesh['mesh_2']['name'], mesh['mesh_2']['nodes']['lat'][:2], mesh['mesh_2']['nodes']['lon'][:2])

    lambda_bsps = compute_ps_bs_gaussians_general(cfg = Config, \
                                          tetra = lambda_bsps['tetra_xyz'], \
                                          event_parameters = ee, \
                                          lambda_bsps      = lambda_bsps, \
                                          vol = lambda_bsps['volumes_elements'], \
                                          mesh = mesh)

    lambda_bsps = compute_ps_bs_gaussians_single_zone(cfg = Config, \
                                          tetra = lambda_bsps['tetra_xyz'], \
                                          event_parameters = ee, \
                                          lambda_bsps      = lambda_bsps, \
                                          vol = lambda_bsps['volumes_elements'], \
                                          mesh = mesh)

    #Update lambda_bsps zoones
    lambda_bsps = update_lambda_bsps_dict(cfg              = Config,
                                          lambda_bsps      = lambda_bsps,
                                          Regionalization  = LongTerm['Regionalization'])

    return lambda_bsps
