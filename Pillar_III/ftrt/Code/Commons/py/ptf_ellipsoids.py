import sys
import numpy as np

def build_ellipsoid_objects(**kwargs):

    event_parameters = kwargs.get('event', 'None')
    Config           = kwargs.get('cfg', 'None')
    args             = kwargs.get('args', 'None')

    ellipse = dict()

    location_ellipse_2d_BS_inn = build_location_ellipsoid_objects(event= event_parameters, \
                                                                  cfg  = Config, \
                                                                  args = args, \
                                                                  type = 'inn', \
                                                                  seismicity_type = 'BS')
    ellipse['location_ellipse_2d_BS_inn'] = location_ellipse_2d_BS_inn

    location_ellipse_2d_BS_out = build_location_ellipsoid_objects(event= event_parameters, \
                                                                  cfg  = Config, \
                                                                  args = args, \
                                                                  type = 'out', \
                                                                  seismicity_type = 'BS')
    ellipse['location_ellipse_2d_BS_out'] = location_ellipse_2d_BS_out

    location_ellipse_2d_PS_inn = build_location_ellipsoid_objects(event= event_parameters, \
                                                                  cfg  = Config, \
                                                                  args = args, \
                                                                  type = 'inn', \
                                                                  seismicity_type = 'PS')
    ellipse['location_ellipse_2d_PS_inn'] = location_ellipse_2d_PS_inn

    location_ellipse_2d_PS_out = build_location_ellipsoid_objects(event= event_parameters, \
                                                                  cfg  = Config, \
                                                                  args = args, \
                                                                  type = 'out', \
                                                                  seismicity_type = 'PS')
    ellipse['location_ellipse_2d_PS_out'] = location_ellipse_2d_PS_out

    return ellipse


def eigsorted(cov):
    '''
    Eigenvalues and eigenvectors of the covariance matrix.
    '''
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]



def cov_ellipse(cov, nstd):
    """
    Source: http://stackoverflow.com/a/12321306/1391441
    """

    vals, vecs = eigsorted(cov)
    print(*vecs[:, 0][::-1])

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    theta = np.arctan2(*vecs[:, 0][::-1])

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)

    return width, height, theta


def build_location_ellipsoid_objects(**kwargs):
    """
    From ellipsedata.m
    % Copyright (c) 2014, Hugo Gabriel Eyherabide, Department of Mathematics
    % and Statistics, Department of Computer Science and Helsinki Institute
    % for Information Technology, University of Helsinki, Finland.
    % All rights reserved.

    !!!! Difference with the original matlab function !!!!
    sigma in this python function is a float
    sigma in matlab is a vector

    """


    Config          = kwargs.get('cfg', 'None')
    ee              = kwargs.get('event', 'None')
    sigma           = kwargs.get('sigma', 'None')
    args            = kwargs.get('args', 'None')
    type            = kwargs.get('type', 'None')
    seismicity_type = kwargs.get('seismicity_type', 'None')



    # number of point of the ellipsoids
    nr_points    = int(Config.get('Settings','nr_points_2d_ellipse'))

    sigma_inn    = Config.get('Settings','nSigma')
    sigma_out    = Config.get('Settings','nSigma')

    if(type == 'inn'):
        sigma = float(sigma_inn)
    else:
        sigma = float(sigma_out)+0.5

    # 2d Covariant matrix, eigenvalues and eignevectors
    if(seismicity_type == 'BS'):
       cov_matrix   = np.array([ [ee['position_BS_sigma_yy']**2, 0], [0, ee['position_BS_sigma_xx']**2] ])
    elif(seismicity_type == 'PS'):
       cov_matrix   = np.array([ [ee['position_PS_sigma_yy']**2, 0], [0, ee['position_PS_sigma_xx']**2] ])
    else:
       print('No seismicity Type found. Exit')
       sys.exit()



    # Center of the ellipse
    center       = (ee['ee_utm'][1],ee['ee_utm'][0])


    PV, PD       = np.linalg.eigh(cov_matrix)

    PV           = np.sqrt(np.diag(PV))

    # Build points of ellipse
    theta        = np.linspace(0,2*np.pi,nr_points)
    elpt         = np.dot(np.transpose(np.array([np.cos(theta), np.sin(theta)])) , PV)
    elpt         = np.dot(elpt, np.transpose(PD))

    # Add uncertainty
    elpt         = elpt * sigma

    # shift to the center
    elpt         = np.transpose(elpt)
    elpt[0]      = elpt[0] + center[0]
    elpt[1]      = elpt[1] + center[1]
    elpt         = np.transpose(elpt)

    return elpt
