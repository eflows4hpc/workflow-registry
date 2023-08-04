import os
import ast
import numpy as np
import folium

from ptf_save import define_save_path
from ptf_save import define_file_names



def make_ptf_figures(**kwargs):


    Config                = kwargs.get('cfg', None)
    args                  = kwargs.get('args', None)
    ee                    = kwargs.get('event_parameters', None)
    ptf                   = kwargs.get('ptf', None)
    ptf_saved_files       = kwargs.get('saved_files', None)

    #if(ee['ptf_status'] == True):
    #    key_fcp_alert_level = 'ptf_fcp_official_level'
    #else:
    #    key_fcp_alert_level = 'matrix_fcp_alert_level'

    save_dict = define_save_path(cfg = Config,args = args, event = ee)

    save_dict = define_file_names(cfg = Config,args = args, event = ee, dictionary=save_dict)

    if(ptf != False):

        ptf_poi_map = make_poi_html_map(cfg                = Config,
                                        args               = args,
                                        event_parameters   = ee,
                                        pois               = ptf['POIs'],
                                        # alert_levels       = ptf['alert_levels'],
                                        save_dict          = save_dict)

    return ptf_saved_files

def make_poi_html_map(**kwargs):

    Config                = kwargs.get('cfg', None)
    args                  = kwargs.get('args', None)
    ee                    = kwargs.get('event_parameters', None)
    pois                  = kwargs.get('pois', None)
    #alert_levels          = kwargs.get('alert_levels', None)
    save_dict             = kwargs.get('save_dict', None)

    method             = ast.literal_eval(Config.get('alert_levels', 'fcp_method'))
    probabilities      = ast.literal_eval(Config.get('alert_levels', 'probabilities'))
    p_levels           = np.arange(probabilities[0],probabilities[1]+1,probabilities[2]) * 0.001
    probability_labels = []
    for i in range(len(p_levels)):
        p_labe = ("1-p%.2f" % p_levels[i])
        probability_labels.append(p_labe)




    ev_depth = ("%.1f" % ee['depth'])
    ev_lat   = ("%.2f" % ee['lat'])
    ev_lon   = ("%.2f" % ee['lon'])
    m16      = ("%.2f" % ee['mag_percentiles']['p16'])
    m50      = ("%.2f" % ee['mag_percentiles']['p50'])
    m84      = ("%.2f" % ee['mag_percentiles']['p84'])
    ot       = ("%s"   % ee['ot'])
    area     = ("%s"   % ee['area'])

    Title    = area + ' Event, OT: ' + ot
    Title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(Title)

    # define map file
    map_file = save_dict['event_path'] + os.sep + save_dict['poi_html_map'] + '.html'

    print(" --> Create html map: ", map_file)


    # define event display parameters
    #event_icona   = f"""<div><svg><rect x="35", y="35" width="20" height="20" fill="blue" opacity="0.6"/></svg></div>"""
    #event_icona   = f"""<div><svg><rect x="35", y="35" width="20" height="20" fill="blue" opacity="0.6"/></svg></div>"""
    event_icona   = f"""<div><svg><circle cx="10" cy="10" r="10" fill="blue" opacity="0.5"/></svg></div>"""
    event_comment = 'Epicenter: Lat: ' + ev_lat + '\nLon: ' + ev_lon + '\nDepth: ' + ev_depth + ' [km]' + '\n' + \
                    'Mag (16-50-84): ' + m16 + ' ' + m50 + ' ' + m84 + '\n' + \
                    'OT: ' + ot
    # Inizialize map
    m = folium.Map(location=[ev_lat, ev_lon], zoom_start=11, tiles="OpenStreetMap")
    m.get_root().html.add_child(folium.Element(Title_html))

    # Add event to map
    folium.Marker(
        location=[ev_lat, ev_lon],
        popup=event_comment,
        icon=folium.DivIcon(html=event_icona),
    ).add_to(m)

    # Add average pois
    #########################################
    average_layer = folium.FeatureGroup(name="average", show=False)
    for i in range(len(pois['selected_pois'])):

        # Poi Definitions
        if(alert_levels['average']['level_type'][i] == 1):
            poi_color = 'green'
        elif(alert_levels['average']['level_type'][i] == 2):
            poi_color = 'yellow'
        elif(alert_levels['average']['level_type'][i] == 3):
            poi_color = 'red'
        else:
            poi_color = 'black'

        hv            = ("%.3e" % alert_levels['average']['level_values'][i])
        poi_comment   = 'Poi: ' + pois['selected_pois'][i]  + '\n' + \
                        'Alert level: ' + alert_levels['average']['level_alert'][i]  + '\n' + \
                        'Hazard Value: ' + hv
        poi_icona     = f"""<div><svg><circle cx="10" cy="10" r="5" fill=""" + poi_color + """ opacity="1"/></svg></div>"""

        folium.Marker(
            location=[pois['selected_lat'][i], pois['selected_lon'][i]],
            popup=poi_comment,
            icon=folium.DivIcon(html=poi_icona),
        ).add_to(average_layer) #m

    # Add best pois
    #########################################
    best_layer = folium.FeatureGroup(name="best", show=True)
    for i in range(len(pois['selected_pois'])):

        # Poi Definitions
        if(alert_levels['best']['level_type'][i] == 1):
            poi_color = 'green'
        elif(alert_levels['best']['level_type'][i] == 2):
            poi_color = 'yellow'
        elif(alert_levels['best']['level_type'][i] == 3):
            poi_color = 'red'
        else:
            poi_color = 'black'

        hv            = ("%.3e" % alert_levels['best']['level_values'][i])
        poi_comment   = 'Poi: ' + pois['selected_pois'][i]  + '\n' + \
                        'Alert level: ' + alert_levels['best']['level_alert'][i]  + '\n' + \
                        'Hazard Value: ' + hv
        poi_icona     = f"""<div><svg><circle cx="10" cy="10" r="5" fill=""" + poi_color + """ opacity="1"/></svg></div>"""

        folium.Marker(
            location=[pois['selected_lat'][i], pois['selected_lon'][i]],
            popup=poi_comment,
            icon=folium.DivIcon(html=poi_icona),
        ).add_to(best_layer) #m

    # Add matrix
    #########################################
    matix_layer = folium.FeatureGroup(name="matrix", show=False)
    for i in range(len(pois['selected_pois'])):

        # Poi Definitions
        if(alert_levels['matrix_poi']['level_alert'][i] == 'information'):
            poi_color = 'green'
        elif(alert_levels['matrix_poi']['level_alert'][i] == 'advisory'):
            poi_color = 'yellow'
        elif(alert_levels['matrix_poi']['level_alert'][i] == 'watch'):
            poi_color = 'red'
        else:
            poi_color = 'black'

        poi_comment   = 'Poi: ' + pois['selected_pois'][i]  + '\n' + \
                        'Alert level: ' + alert_levels['matrix_poi']['level_alert'][i]
        poi_icona     = f"""<div><svg><circle cx="10" cy="10" r="5" fill=""" + poi_color + """ opacity="1"/></svg></div>"""

        folium.Marker(
            location=[pois['selected_lat'][i], pois['selected_lon'][i]],
            popup=poi_comment,
            icon=folium.DivIcon(html=poi_icona),
        ).add_to(matix_layer) #m

    ###
    #########################################
    for j in range(len(probability_labels)):

      #.05 .5 .95
      if j == 0 or j == 9 or j == 18:

        pp_layer = folium.FeatureGroup(name=probability_labels[j], show=False)
        for i in range(len(pois['selected_pois'])):

            # Poi Definitions
            if(alert_levels['probability']['level_type'][i,j] == 1):
                poi_color = 'green'
            elif(alert_levels['probability']['level_type'][i,j] == 2):
                poi_color = 'yellow'
            elif(alert_levels['probability']['level_type'][i,j] == 3):
                poi_color = 'red'
            else:
                poi_color = 'black'

            hv            = ("%.3e" % alert_levels['probability']['level_values'][i,j])
            poi_comment   = 'Poi: ' + pois['selected_pois'][i]  + '\n' + \
                            'Alert level: ' + alert_levels['probability']['level_alert'][i][j] + '\n' + \
                            'Hazard Value: ' + hv
            poi_icona     = f"""<div><svg><circle cx="10" cy="10" r="5" fill=""" + poi_color + """ opacity="1"/></svg></div>"""

            folium.Marker(
                location=[pois['selected_lat'][i], pois['selected_lon'][i]],
                popup=poi_comment,
                icon=folium.DivIcon(html=poi_icona),
            ).add_to(pp_layer) #m

        pp_layer.add_to(m)


    average_layer.add_to(m)
    best_layer.add_to(m)
    matix_layer.add_to(m)
    folium.LayerControl(name='Hazard cut curve types').add_to(m)

    # Save html map
    m.save(outfile = map_file)

    return True

