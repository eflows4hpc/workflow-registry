
import numpy as np

from operator          import itemgetter
from ptf_mix_utilities import ray_tracing_method


def pre_selection_of_scenarios(**kwargs):


    Config            = kwargs.get('cfg', 'None')
    args              = kwargs.get('args', 'None')
    ee                = kwargs.get('event_parameters', 'None')
    LongTermInfo      = kwargs.get('LongTermInfo', 'None')
    PSBarInfo         = kwargs.get('PSBarInfo', 'None')
    ellipses          = kwargs.get('ellipses', 'None')
    ellipse_2d_BS_inn = kwargs.get('ellipse_2d_BS_inn', 'None')
    ellipse_2d_BS_out = kwargs.get('ellipse_2d_BS_out', 'None')
    ellipse_2d_PS_inn = kwargs.get('ellipse_2d_PS_inn', 'None')
    ellipse_2d_PS_out = kwargs.get('ellipse_2d_PS_out', 'None')


    pre_selection = dict()



    pre_selection = pre_selection_magnitudes(cfg              = Config, \
                                             event_parameters = ee, \
                                             pre_selection    = pre_selection, \
                                             PS_mag           = LongTermInfo['Discretizations']['PS-1_Magnitude'], \
                                             BS_mag           = LongTermInfo['Discretizations']['BS-1_Magnitude'])

    ### Check if estimated magnitudes +- uncertainties are can be found in scenarios, if not do not pre-selected
    pre_selection = check_mag_for_pre_selection(event_parameters = ee,
                                                pre_selection    = pre_selection, \
                                                PS_mag           = LongTermInfo['Discretizations']['PS-1_Magnitude'], \
                                                BS_mag           = LongTermInfo['Discretizations']['BS-1_Magnitude'])


    if(pre_selection['BS_scenarios'] == False and pre_selection['PS_scenarios'] == False):
        #pre_selection['BS2_Position_Selection_out'] = False
        #pre_selection['apply_decision_matrix']      = True
        #ee['apply_decision_matrix']                 = True
        #pre_selection['applay_ptf']                 = False
        #ee['applay_ptf']                            = False
        print(" --> No scenarios for this event. Apply Decision Matrix")
        return False

    if(pre_selection['BS_scenarios'] == True):
        pre_selection = pre_selection_BS2_position(cfg                 = Config, \
                                               event_parameters    = ee, \
                                               pre_selection       = pre_selection, \
                                               BS2_pos             = LongTermInfo['Discretizations']['BS-2_Position'], \
                                               ellipse_2d_inn      = ellipses['location_ellipse_2d_BS_inn'], \
                                               ellipse_2d_out      = ellipses['location_ellipse_2d_BS_out'])
    else:
        pre_selection['BS2_Position_Selection_inn'] = np.array([])

    if(pre_selection['PS_scenarios'] == True):
        pre_selection = pre_selection_PS2_position(cfg                 = Config, \
                                               event_parameters    = ee, \
                                               pre_selection       = pre_selection, \
                                               PS2_pos             = LongTermInfo['Discretizations']['PS-2_PositionArea'], \
                                               ellipse_2d_inn      = ellipses['location_ellipse_2d_PS_inn'], \
                                               ellipse_2d_out      = ellipses['location_ellipse_2d_PS_out'])

        pre_selection = pre_selection_Bar_PS_Model(cfg                 = Config,\
                                               event_parameters    = ee, \
                                               pre_selection       = pre_selection, \
                                               PS2_pos             = LongTermInfo['Discretizations']['PS-2_PositionArea'], \
                                               BarPSperModel       = PSBarInfo['BarPSperModel'], \
                                               ellipse_2d_inn      = ellipses['location_ellipse_2d_PS_inn'], \
                                               ellipse_2d_out      = ellipses['location_ellipse_2d_PS_out'])


    return pre_selection

def check_mag_for_pre_selection(**kwargs):

    ee            = kwargs.get('event_parameters', 'None')
    PS_mag        = kwargs.get('PS_mag', 'None')
    BS_mag        = kwargs.get('BS_mag', 'None')
    pre_selection = kwargs.get('pre_selection', 'None')

    if(ee['mag_percentiles']['p84'] < BS_mag['Val'][0] or ee['mag_percentiles']['p16'] > BS_mag['Val'][-1]):
        pre_selection['BS_scenarios'] = False
        print(" --> Magnitude event outside Magnitide BS scenarios -->", pre_selection['BS_scenarios'])
    else:
        pre_selection['BS_scenarios'] = True
    if(ee['mag_percentiles']['p84'] < PS_mag['Val'][0] or ee['mag_percentiles']['p16'] > PS_mag['Val'][-1]):
        pre_selection['PS_scenarios'] = False
        print(" --> Magnitude event outside Magnitide PS scenarios -->", pre_selection['BS_scenarios'])

    else:
        pre_selection['PS_scenarios'] = True

    return pre_selection

def pre_selection_Bar_PS_Model(**kwargs):
    """
    This function uses a ray tracing method decorated with numba
    """

    Config         = kwargs.get('cfg', 'None')
    ee             = kwargs.get('event_parameters', 'None')
    pre_selection  = kwargs.get('pre_selection', 'None')
    PS2_pos        = kwargs.get('PS2_pos', 'None')
    BarPSperModel  = kwargs.get('BarPSperModel', 'None')
    ellipse_2d_inn = kwargs.get('ellipse_2d_inn', 'None')
    ellipse_2d_out = kwargs.get('ellipse_2d_out', 'None')


    Selected_PS_Mag_idx = pre_selection['sel_PS_Mag_idx'][0]
    print(" --> Index of PS_mag selection:  ", Selected_PS_Mag_idx)



    test_dict = {}

    for i1 in range(len(Selected_PS_Mag_idx)):
        imag = Selected_PS_Mag_idx[i1]


        for imod in range(len(BarPSperModel[imag])):

            if('utm_pos_lat' in BarPSperModel[imag][imod]):

                if(BarPSperModel[imag][imod]['utm_pos_lat'].size >=2):
                    points     = zip(BarPSperModel[imag][imod]['utm_pos_lon'], BarPSperModel[imag][imod]['utm_pos_lat'])
                    inside_inn = [ray_tracing_method(point[0], point[1], ellipse_2d_inn) for point in points]

                elif(BarPSperModel[imag][imod]['utm_pos_lat'].size ==1 ):
                    inside_inn = ray_tracing_method(BarPSperModel[imag][imod]['utm_pos_lon'][0], BarPSperModel[imag][imod]['utm_pos_lat'][0], ellipse_2d_inn)

                else:
                    pass

                Inside_in_BarPSperModel = {'inside' : np.where(inside_inn)[0]}
                test_dict.setdefault(imag, {})[imod] = Inside_in_BarPSperModel


    pre_selection['Inside_in_BarPSperModel'] = test_dict

    return pre_selection

def pre_selection_PS2_position(**kwargs):
    """
    This function uses a ray tracing method decorated with cumba
    """

    #from shapely.geometry import Point, MultiPoint
    #from shapely.geometry.polygon import Polygon
    #import matplotlib.path as mpltPath

    Config         = kwargs.get('cfg', 'None')
    ee             = kwargs.get('event_parameters', 'None')
    pre_selection  = kwargs.get('pre_selection', 'None')
    PS2_pos        = kwargs.get('PS2_pos', 'None')
    ellipse_2d_inn = kwargs.get('ellipse_2d_inn', 'None')
    ellipse_2d_out = kwargs.get('ellipse_2d_out', 'None')


    # https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    #start_time = time()
    #print("--------------", ellipse_2d_inn)
    points     = zip(PS2_pos['utm_y'], PS2_pos['utm_x'])
    inside_inn = [ray_tracing_method(point[0], point[1], ellipse_2d_inn) for point in points]

    points     = zip(PS2_pos['utm_y'], PS2_pos['utm_x'])
    inside_out = [ray_tracing_method(point[0], point[1], ellipse_2d_out) for point in points]

    # Map common indices
    bool_array       = np.in1d(np.where(inside_out)[0], np.where(inside_inn)[0])
    common_positions = np.where(bool_array)[0]

    # fill dictionary
    pre_selection['PS2_Position_Selection_inn']    = np.where(inside_inn)[0]
    pre_selection['PS2_Position_Selection_out']    = np.where(inside_out)[0]
    pre_selection['PS2_Position_Selection_common'] = np.take(pre_selection['PS2_Position_Selection_out'],common_positions)

    print(" --> PS2_Position inner:         %4d positions found" % (len(pre_selection['PS2_Position_Selection_inn'])))
    print(" --> PS2_Position outer:         %4d positions found" % (len(pre_selection['PS2_Position_Selection_out'])))
    print(" --> PS2_Position inn and out:   %4d positions found" % (len(pre_selection['PS2_Position_Selection_common'])))
    #print ("Ray Tracing Elapsed time: " + str(time()-start_time))


    return pre_selection

def pre_selection_BS2_position(**kwargs):
    """
    This function uses a ray tracing method decorated with cumba
    """

    #from shapely.geometry import Point, MultiPoint
    #from shapely.geometry.polygon import Polygon
    #import matplotlib.path as mpltPath

    Config         = kwargs.get('cfg', 'None')
    ee             = kwargs.get('event_parameters', 'None')
    pre_selection  = kwargs.get('pre_selection', 'None')
    BS2_pos        = kwargs.get('BS2_pos', 'None')
    ellipse_2d_inn = kwargs.get('ellipse_2d_inn', 'None')
    ellipse_2d_out = kwargs.get('ellipse_2d_out', 'None')


    # https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    #start_time = time()
    points     = zip(BS2_pos['utm_y'], BS2_pos['utm_x'])
    inside_inn = [ray_tracing_method(point[0], point[1], ellipse_2d_inn) for point in points]

    points     = zip(BS2_pos['utm_y'], BS2_pos['utm_x'])
    inside_out = [ray_tracing_method(point[0], point[1], ellipse_2d_out) for point in points]

    # Map common indices
    bool_array       = np.in1d(np.where(inside_out)[0], np.where(inside_inn)[0])
    common_positions = np.where(bool_array)[0]

    # fill dictionary
    #print("-------------------------", inside_inn)
    #print("-------------------------", np.where(inside_inn)[0])
    pre_selection['BS2_Position_Selection_inn']    = np.where(inside_inn)[0]
    pre_selection['BS2_Position_Selection_out']    = np.where(inside_out)[0]
    pre_selection['BS2_Position_Selection_common'] = np.take(pre_selection['BS2_Position_Selection_out'],common_positions)

    print(" --> BS2_Position inner:         %4d positions found" % (len(pre_selection['BS2_Position_Selection_inn'])))
    print(" --> BS2_Position outer:         %4d positions found" % (len(pre_selection['BS2_Position_Selection_out'])))
    print(" --> BS2_Position inn and out:   %4d positions found" % (len(pre_selection['BS2_Position_Selection_common'])))
    #print ("Ray Tracing Elapsed time: " + str(time()-start_time))

    return pre_selection

def pre_selection_magnitudes(**kwargs):

    Config        = kwargs.get('cfg', 'None')
    ee            = kwargs.get('event_parameters', 'None')
    PS_mag        = kwargs.get('PS_mag', 'None')
    BS_mag        = kwargs.get('BS_mag', 'None')
    pre_selection = kwargs.get('pre_selection', 'None')

    ################################################################
    # Some Variables
    nSigma     = float(Config.get('Settings', 'nSigma'))
    max_BS_mag = float(Config.get('Settings', 'Mag_BS_Max'))

    val_PS         = np.array(PS_mag['Val'])
    ID_PS          = list(PS_mag['ID'])
    val_BS         = np.array(BS_mag['Val'])
    ID_BS          = list(BS_mag['ID'])

    ################################################################
    # Magnitude range given by sigma
    min_mag  = ee['mag'] - ee['MagSigma'] * nSigma
    max_mag  = ee['mag'] + ee['MagSigma'] * nSigma

    # Due problemai da risolvere: se magnitudo inferiore uscire perche no allerta --> solo matrice decisionale
    #                             Se magnitudo superiore a quelle previste, diventa la massima prevista
    # Potrebbe essere:
    # se max_mag <= val_PS[0] e val_BS[0]--> uscire
    # Se mag_min >= val_PS[-1] e val_BS[-1] --> solo ultimo
    # Altrimenti range in mezzo
    # Ps selection


    if(max_mag <= val_PS[0]):
        sel_PS_Mag_val = np.array([])
        sel_PS_Mag_idx = (sel_PS_Mag_val,)
        sel_PS_Mag_IDs = []

    elif(min_mag >= val_PS[-1]):
        sel_PS_Mag_val = np.array([val_PS[-1]])
        sel_PS_Mag_idx = (sel_PS_Mag_val,)
        sel_PS_Mag_IDs = [sel_PS_Mag_idx[-1]]

    else:
        sel_PS_Mag_val = val_PS[(val_PS >= min_mag) & (val_PS <= max_mag)]
        sel_PS_Mag_idx = np.where((val_PS >= min_mag) & (val_PS <= max_mag))
        # To fix if mag uncertainty too small for val_PS element intervals
        # Find closest magnitude
        if(len(sel_PS_Mag_idx[0]) == 0):
            idx = np.array((np.abs(val_PS-max_mag)).argmin())
            sel_PS_Mag_IDs = list(itemgetter(idx)(ID_PS))
        else:
            sel_PS_Mag_IDs = list(itemgetter(*sel_PS_Mag_idx[0])(ID_PS))


    # BS

    if(max_mag <= val_BS[0]):
        sel_BS_Mag_val = np.array([])
        sel_BS_Mag_idx = (sel_BS_Mag_val,)
        sel_BS_Mag_IDs = []

    elif(min_mag >= val_BS[-1]):
        # sel_BS_Mag_val = np.array([val_BS[-1]])
        sel_BS_Mag_val = np.array([])
        sel_BS_Mag_idx = (sel_BS_Mag_val,)
        sel_BS_Mag_IDs = []
        # non devo prendere nulla
        #sel_BS_Mag_IDs = [sel_BS_Mag_idx[-1]]



    else:
        sel_BS_Mag_val = val_BS[(val_BS >= min_mag) & (val_BS <= max_mag)]
        sel_BS_Mag_idx = np.where((val_BS >= min_mag) & (val_BS <= max_mag))
        if(len(sel_BS_Mag_idx[0]) == 0):
            idx = np.array((np.abs(val_BS-max_mag)).argmin())
            sel_BS_Mag_IDs = list(itemgetter(idx)(ID_BS))
        else:
            sel_BS_Mag_IDs = list(itemgetter(*sel_BS_Mag_idx[0])(ID_BS))

    #print(sel_BS_Mag_val, sel_BS_Mag_idx)

    pre_selection['sel_PS_Mag_val'] = sel_PS_Mag_val
    pre_selection['sel_PS_Mag_idx'] = sel_PS_Mag_idx
    pre_selection['sel_PS_Mag_IDs'] = sel_PS_Mag_IDs
    pre_selection['sel_BS_Mag_val'] = sel_BS_Mag_val
    pre_selection['sel_BS_Mag_idx'] = sel_BS_Mag_idx
    pre_selection['sel_BS_Mag_IDs'] = sel_BS_Mag_IDs

    print(" --> PS magnitues values:       ", *sel_PS_Mag_val)
    print(" --> PS magnitues ID:           ", *sel_PS_Mag_IDs)
    print(" --> BS magnitues values:       ", *sel_BS_Mag_val)
    print(" --> BS magnitues ID:           ", *sel_BS_Mag_IDs)

    return pre_selection
