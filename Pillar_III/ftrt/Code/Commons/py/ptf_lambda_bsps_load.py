import os
import ast
import utm
import math

import scipy.interpolate
import scipy.io
import scipy.spatial

import numpy as np

from scipy.stats import norm
from scipy.stats import distributions


def load_lambda_BSPS(**kwargs):


    Config   = kwargs.get('cfg', 'None')
    args     = kwargs.get('args', 'None')
    ee_d     = kwargs.get('event_parameters', 'None')
    longTerm = kwargs.get('LongTermInfo', 'None')

    ################################################################
    # Some Variables
    nSigma = float(Config.get('Settings','nSigma'))



    d = dict()

    ################################################################
    ## Basic mean and covariant matrix from location parameters
    """
    d['ee_PosCovMat_2d'] = np.array([[ee_d['cov_matrix']['XX'], ee_d['cov_matrix']['XY']], \
                                     [ee_d['cov_matrix']['XY'], ee_d['cov_matrix']['YY']]])

    d['PosMean_2d']      = np.array([ee_d['ee_utm'][1], \
                                     ee_d['ee_utm'][0]])

    d['PosCovMat_3d']    = np.array([[ee_d['cov_matrix']['XX'], ee_d['cov_matrix']['XY'], ee_d['cov_matrix']['XZ']], \
                                     [ee_d['cov_matrix']['XY'], ee_d['cov_matrix']['YY'], ee_d['cov_matrix']['YZ']], \
                                     [ee_d['cov_matrix']['XZ'], ee_d['cov_matrix']['YZ'], ee_d['cov_matrix']['ZZ']]])

    d['PosMean_3d']      = np.array([ee_d['ee_utm'][0], \
                                     ee_d['ee_utm'][1], \
                                     ee_d['depth'] * 1000.0])
    """

    ################################################################
    ## Variables for lambda
    d['lambdaBSPS']                                = {}

    d['lambdaBSPS']['hypo_utm']                    = np.array([ee_d['ee_utm'][0], \
                                                               ee_d['ee_utm'][1], \
                                                               ee_d['depth'] ])

    d['lambdaBSPS']['utmzone_hypo']                = ee_d['ee_utm'][2]

    d['lambdaBSPS']['NormCov']                     = ee_d['PosCovMat_3d']

    d['lambdaBSPS']['confid_lev']                  = norm.cdf(nSigma) - norm.cdf(-1 * nSigma)

    d['lambdaBSPS']['dchi2']                       = distributions.chi2.ppf(d['lambdaBSPS']['confid_lev'], 3)

    d['lambdaBSPS']['SD']                          = math.sqrt(d['lambdaBSPS']['dchi2'])

    d['lambdaBSPS']['mesh']                        = get_meshes(event_parameters=ee_d, cfg=Config)

    d['lambdaBSPS']['covariance_epicenter_volume'] = get_cov_volume(cov_matrix=ee_d['PosCovMat_3d'], \
                                                                    std=d['lambdaBSPS']['SD'])

    d['lambdaBSPS']['npts_mw']                     = get_npts_mw(volume=d['lambdaBSPS']['covariance_epicenter_volume'], \
                                                                 cfg=Config)

    d['lambdaBSPS']['gaussian_ellipsoid']          = get_gaussian_ellipsoid_3d(event_parameters=ee_d, \
                                                                               cov_matrix=ee_d['PosCovMat_3d'], \
                                                                               std=d['lambdaBSPS']['SD'], \
                                                                               npts=d['lambdaBSPS']['npts_mw'])

    d['lambdaBSPS']                                = get_gaussian_ellipsoid_tetraedons(ellipsoid=d['lambdaBSPS'], \
                                                                                   event_parameters=ee_d)


    return d['lambdaBSPS']

def get_gaussian_ellipsoid_tetraedons(**kwargs):
    """
    Fondamentalmente tutta questa funzione è presa dal codice già tradotto da R. Tonini
    """

    el   = kwargs.get('ellipsoid', None)
    ee   = kwargs.get('event_parameters', None)

    xp = el['gaussian_ellipsoid']['xp']
    yp = el['gaussian_ellipsoid']['yp']
    zp = el['gaussian_ellipsoid']['zp'] #*-1.0

    #all= np.concatenate((xp,yp,zp), axis=0)
    #print(np.shape(all))

    #(vertices,ia,ic) = np.unique(all,return_index=True,return_inverse=True, axis=0)
    # array preparation for creating tetrahedrons
    # This il like sss on matptf
    sss = np.vstack([xp.flatten(), yp.flatten(), zp.flatten()]).transpose()
    points_xyz = np.unique(sss, axis=0)
    n_points, tmp = np.shape(points_xyz)

    # lon lat conversion
    points_ll = np.zeros((n_points, 3))
    for i in range(n_points):
        points_ll[i, [1, 0]] = utm.to_latlon(points_xyz[i, 0],
                                             points_xyz[i, 1],
                                             ee['ee_utm'][2], ee['ee_utm'][3])

    points_ll[:, 2] = points_xyz[:, 2]

    # Good but slower (0.0030319690704345703 <=> 0.0013072490692138672)
    #from pyhull.delaunay import DelaunayTri
    #tetrahedrons = np.asarray(DelaunayTri(points_ll).vertices)
    #t1 = time.time()
    #print(t1-t0)
    # tetrahedron discretization (based on the points on the surface)
    tessellation = scipy.spatial.Delaunay(points_ll)
    tetrahedrons = tessellation.simplices


    # computing barycenters
    tetra_bar          = {}
    tetra_bar["utm_x"] = np.mean(points_xyz[tetrahedrons, 0], axis=1)
    tetra_bar["utm_y"] = np.mean(points_xyz[tetrahedrons, 1], axis=1)
    tetra_bar["lon"]   = np.mean(points_ll[tetrahedrons, 0], axis=1)
    tetra_bar["lat"]   = np.mean(points_ll[tetrahedrons, 1], axis=1)
    tetra_bar["depth"] = np.mean(points_ll[tetrahedrons, 2], axis=1)

    tetra_xyz = np.column_stack((tetra_bar["utm_x"],
                                 tetra_bar["utm_y"],
                                 tetra_bar["depth"]))

    n_tetra = len(tetra_bar["lon"])
    print(" --> N. Tetra of Gaus. Ell.:     {0}".format(n_tetra))

    # computing tetrahedrons volume
    volume = np.zeros((n_tetra))
    for i in range(n_tetra):
        mm = np.column_stack((points_xyz[tetrahedrons[i, :], :],
                              np.array([1, 1, 1, 1])))
        volume[i] = np.abs(np.linalg.det(mm)/6.)

    volume_tot = np.sum(volume)
    print(" --> Volume of Tetra Gaus. Ell.: %.8e [m^3]" % volume_tot)

    Vol_diff_perc = (el['gaussian_ellipsoid']['vol'] - volume_tot) / el['gaussian_ellipsoid']['vol']*100
    print(" --> Volume diff. Gaus. <--> Tetra: %.2f [%%]" % Vol_diff_perc)

    el['tetra_bar']                 = tetra_bar
    el['tetrahedrons']              = tetrahedrons
    el['gaussian_ellipsoid_volume'] = volume_tot
    el['volumes_elements']          = volume
    el['tetra_xyz']                 = tetra_xyz
    #print("---> VOLLLLL", volume)

    return el

def get_gaussian_ellipsoid_3d(**kwargs):

    ee     = kwargs.get('event_parameters', None)
    cov    = kwargs.get('cov_matrix', None)
    std    = kwargs.get('std', None)
    npts   = kwargs.get('npts', None)

    center = [ee['ee_utm'][0], ee['ee_utm'][1], ee['depth']*-1000.0]

    cov = cov*1e6
    w, v = np.linalg.eigh(cov)
    if np.any(w < 0):
        print('warning: negative eigenvalues')
        w = max(w,0)
    w      = std * np.sqrt(w)    #get std of the cov matrix

    volume = (4./3.) * np.pi * w[0] * w[1] * w[2]


    # Make 3x 11x11 arrays
    x, y, z = create_sphere(n_points=npts)


    x = np.transpose(x)
    y = np.transpose(y)
    z = np.transpose(z)

    # Flattern 11x11 array
    ap = np.array([np.ravel(x), np.ravel(y), np.ravel(z)])

    #Arrivato qui, ora traslare verso il centro dell=a matrice di covarianza
    bp = np.dot(np.dot(v, np.diag(w)), ap) +  \
         np.transpose(np.tile(center, (np.shape(ap)[1], 1)))

    xp = np.reshape(bp[0, :], np.shape(x))
    yp = np.reshape(bp[1, :], np.shape(y))
    zp = np.reshape(bp[2, :], np.shape(z))

    ellipsoid = {'xp':xp, 'yp':yp, 'zp':zp, 'vol': volume}
    print(" --> Volume of Gaus. Ell.:       %.8e [m^3]" % volume)

    return ellipsoid

def create_sphere(n_points=None, radius=None):
    """
    Create a discrete 3D spheric surface (points)
    Reference to create the shere:
       https://it.mathworks.com/matlabcentral/answers/48240-surface-of-a-equation:
       n = 100;
        r = 1.5;
        theta = (-n:2:n)/n*pi;
        phi = (-n:2:n)'/n*pi/2;
        cosphi = cos(phi); cosphi(1) = 0; cosphi(n+1) = 0;
        sintheta = sin(theta); sintheta(1) = 0; sintheta(n+1) = 0;
        x = r*cosphi*cos(theta);
        y = r*cosphi*sintheta;
        z = r*sin(phi)*ones(1,n+1);
        surf(x,y,z)
        xlabel('X'); ylabel('Y'); zlabel('Z')
    """
    if radius is None:
        radius = 1.0

    if n_points is None:
        n_points = 20

    theta = np.matrix(np.arange(-1*n_points,n_points+1,2) / n_points * np.pi)
    phi   = np.matrix(np.arange(-1*n_points,n_points+1,2) / n_points * np.pi / 2)
    phi   = phi.transpose()

    X = radius*np.matmul(np.cos(phi),np.cos(theta))
    Y = radius*np.matmul(np.cos(phi),np.sin(theta))
    Z = radius*np.matmul(np.sin(phi),np.matrix(np.ones(11)))

    # Set to 0 the very small numbers
    X[0] = 0
    X[-1] = 0
    Y[0] = 0
    Y[-1] = 0
    Y[:,0] = 0
    Y[:,-1] = 0

    return X,Y,Z

def get_npts_mw(**kwargs):
    """
    Calculate the number of points to define de ellipsoide.
    Fitting function a*x**b found by F.Romano
    """
    volume = kwargs.get('volume', None)
    Config = kwargs.get('cfg', None)

    a         = float(Config.get('lambda','a'))
    b         = float(Config.get('lambda','b'))
    n_tetra   = float(Config.get('lambda','Ntetra'))
    vol_tetra = float(Config.get('lambda','Vol_tetra'))

    npts_mw   = np.ceil(a*(volume*n_tetra/vol_tetra)**b).astype(int)

    npts_mw   = max(10, npts_mw)

    return npts_mw


def get_cov_volume(**kwargs):

    cov_matrix = kwargs.get('cov_matrix', None)
    std        = kwargs.get('std', None)

    w, v = np.linalg.eig(cov_matrix)
    l_major = std*np.sqrt(w[0]) * 1000.0
    l_inter = std*np.sqrt(w[1]) * 1000.0
    l_minor = std*np.sqrt(w[2]) * 1000.0



    volume = (4./3.) * np.pi * l_major * l_inter * l_minor

    return volume

def get_meshes(**kwargs):

    ee      = kwargs.get('event_parameters', None)
    Config  = kwargs.get('cfg', None)

    path  = Config.get('lambda','mesh_path')
    faces = ast.literal_eval(Config.get('lambda','mesh_faces'))
    nodes = ast.literal_eval(Config.get('lambda','mesh_nodes'))
    names = ast.literal_eval(Config.get('lambda','mesh_names'))



    mesh_d = dict()

    for i in range(len(faces)):


        f = os.path.join(path, faces[i])
        n = os.path.join(path, nodes[i])



        mesh_name                      = names[i]
        mesh_d[mesh_name]              = dict()
        mesh_d[mesh_name]["faces"]     = np.loadtxt(f)
        mesh_d[mesh_name]["nodes"]     = np.loadtxt(n)
        mesh_d[mesh_name]["nodes_utm"] = utm.from_latlon(mesh_d[mesh_name]["nodes"][:, 1],
                                                         mesh_d[mesh_name]["nodes"][:, 2],
                                                         ee['ee_utm'][2])

    return mesh_d

##################################################
##################################################

def get_sd(confidence=None, degree_of_freedom=None):
    """
    calculate a sort of standard deviation, using the chi2 cumulative
    distribution function (cdf) with a given level of confidence
    and a given degrees of freedom
    """
    if confidence is None:
        confidence = 0.68

    if degree_of_freedom is None:
        degree_of_freedom = 3

    return np.sqrt(scipy.stats.chi2.ppf(confidence, degree_of_freedom))




def get_moho(path, ll2utm):
    """
    """
    moho = {}
    moho["lon"], moho["lat"], moho["depth"] = np.loadtxt(path, unpack=True)
    tmp = ll2utm(moho["lon"], moho["lat"])
    moho["utm_x"] = tmp[0]
    moho["utm_y"] = tmp[1]
    moho["depth"] = -moho["depth"]*1000.


    return moho





# input data
#home = os.path.expanduser("~")
"""
home = os.path.join("/", "work", "tonini")
dir_data = os.path.join(home, "tsumaps", "lambda_separation_BS_PS")
dir_mesh = os.path.join(home, "tsumaps", "lambda_separation_BS_PS", "mesh")
file_hypocenter = os.path.join(dir_data, "crete_eq_hypo.dat")
file_covariance = os.path.join(dir_data, "cov.dat")
file_moho = os.path.join(dir_data, "Grid025_MOHO_FIXED.xyz")
mesh_names = ["HeA", "CaA", "CyA"]

# get EarlyEst hypocenter
#hypocentre = get_ee_hypocentre(file_hypocenter)
hypocentre = get_ee_hypocentre()
print(hypocentre)

# coordinate transformation
ll2utm = pyproj.Proj(proj='utm',
                     zone=hypocentre["utm_zone_number"],
                     ellps='WGS84')

# get EarlyEst covariance matrix in km^2 and convert in m^2
covariance = np.loadtxt(file_covariance)
covariance = covariance * 10**6
# print(covariance)

# calculating a sort of standard deviation
sd = get_sd()

# get moho
moho = get_moho(file_moho, ll2utm)

# get meshes
meshes = get_meshes(dir_mesh, mesh_names, ll2utm)

# compute lambda BS/PS
lambda_bs, lambda_ps = get_lambda_bs_ps(hypocentre, covariance, sd,
                                        moho, meshes, dir_data)
"""


def get_ellipsoide_volume(covariance, standard_deviation):
    """
    calculate volume of the ellipsoide corresponding to the
    covariance matrix
    """

    w, v = np.linalg.eig(covariance)

    # ellipsoide semi-axis in km
    l_major = standard_deviation*np.sqrt(w[0])
    l_inter = standard_deviation*np.sqrt(w[1])
    l_minor = standard_deviation*np.sqrt(w[2])
    return (4./3.) * np.pi * l_major * l_inter * l_minor
