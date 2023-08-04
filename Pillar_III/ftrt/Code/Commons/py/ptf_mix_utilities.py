import utm
import math
import numpy as np
import scipy

#from numba import jit
import numpy.matlib as npm
import numbers



def merge_event_dictionaries(**kwargs):

    event_ttt = kwargs.get('event_ttt', None)
    event_ptf = kwargs.get('event_ptf', None)

    full_event = {**event_ttt, **event_ptf}

    return full_event

def check_if_neam_event(**kwargs):
    dictionary       = kwargs.get('dictionary', None)
    cfg              = kwargs.get('cfg', None)

    area_neam        = eval(cfg.get('bounds','neam'))

    inneam = ray_tracing_method(float(dictionary['lon']), float(dictionary['lat']), area_neam)

    if(inneam == True):
        dictionary['inneam'] = True
    else:
        dictionary['inneam'] = False
    return dictionary


def st_to_float(x):

    # if any number
    if isinstance(x,numbers.Number):
        return x
    # if non a number try convert string to float or it
    for type_ in (float):
        try:
            return type_(x)
        except ValueError:
            continue

#@jit(nopython=False, cache=False, fastmath=True)
def ccc(h,s,m):
    ccde       = 1 - scipy.stats.lognorm.cdf(h, s, scale=np.exp(m)).transpose()
    return ccde

#@jit(nopython=True, cache=True)
def ray_tracing_method(x,y,poly):

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


def conversion_to_utm(**kwargs):

    long = kwargs.get('longTerm', None)
    pois = kwargs.get('Poi',      None)
    ee   = kwargs.get('event',    None)
    PSBa = kwargs.get('PSBarInfo',    None)

    a = utm.from_latlon(np.array(long['Discretizations']['BS-2_Position']['Val_y']), np.array(long['Discretizations']['BS-2_Position']['Val_x']), ee['ee_utm'][2])
    long['Discretizations']['BS-2_Position']['utm_y']   = a[1]
    long['Discretizations']['BS-2_Position']['utm_x']   = a[0]
    long['Discretizations']['BS-2_Position']['utm_nr']  = a[2]
    long['Discretizations']['BS-2_Position']['utm_reg'] = a[3]

    a = utm.from_latlon(np.array(long['Discretizations']['PS-2_PositionArea']['Val_y']), np.array(long['Discretizations']['PS-2_PositionArea']['Val_x']), ee['ee_utm'][2])
    long['Discretizations']['PS-2_PositionArea']['utm_y']   = a[1]
    long['Discretizations']['PS-2_PositionArea']['utm_x']   = a[0]
    long['Discretizations']['PS-2_PositionArea']['utm_nr']  = a[2]
    long['Discretizations']['PS-2_PositionArea']['utm_reg'] = a[3]

    a = utm.from_latlon(np.array(pois['lat']), np.array(pois['lon']), ee['ee_utm'][2])
    pois['utm_lat'] = a[1]
    pois['utm_lon'] = a[0]
    pois['utm_nr']  = a[2]
    pois['utm_reg'] = a[3]

    for i in range(len(PSBa['BarPSperModel'])):
        for j in range(len(PSBa['BarPSperModel'][i])):
            #print(type(PSBa['BarPSperModel'][i][j]['pos_yy']))
            #sys.exit()
            if PSBa['BarPSperModel'][i][j]['pos_yy'].size < 1:
                PSBa['BarPSperModel'][i][j]['utm_pos_lat'] = np.array([])
                PSBa['BarPSperModel'][i][j]['utm_pos_lon'] = np.array([])
                PSBa['BarPSperModel'][i][j]['utm_pos_nr'] = np.array([])
                PSBa['BarPSperModel'][i][j]['utm_pos_reg'] = np.array([])
                pass
            else:
                a = utm.from_latlon(np.array(PSBa['BarPSperModel'][i][j]['pos_yy']), np.array(PSBa['BarPSperModel'][i][j]['pos_xx']), ee['ee_utm'][2])
                PSBa['BarPSperModel'][i][j]['utm_pos_lat'] = a[0]
                PSBa['BarPSperModel'][i][j]['utm_pos_lon'] = a[1]
                PSBa['BarPSperModel'][i][j]['utm_pos_nr']  = a[2]
                PSBa['BarPSperModel'][i][j]['utm_pos_reg'] = a[3]
            #print("+++++++", i,j,PSBa['BarPSperModel'][i][j]['utm_pos_lat'].size, PSBa['BarPSperModel'][i][j]['utm_pos_lon'].size)

    #print(PSBa['BarPSperModel'][2][1]['utm_pos_lat'].size)

    return long, pois, PSBa

def iterdict(d):
    for k,v in d.items():
        if isinstance(v, dict):
            iterdict(v)
        else:
            #print (k,":",v)
            print(k)

def NormMultiDvec(**kwargs):

    """
    # Here mu and sigma, already inserted into ee dictionary
    # Coordinates in utm
    mu = tmpmu =PosMean_3D = [EarlyEst.lonUTM,EarlyEst.latUTM,EarlyEst.Dep*1.E3]
    Sigma = tmpCOV = EarlyEst.PosCovMat_3D = [EarlyEst.PosSigmaXX EarlyEst.PosSigmaXY EarlyEst.PosSigmaXZ; ...
                         EarlyEst.PosSigmaXY EarlyEst.PosSigmaYY EarlyEst.PosSigmaYZ; ...
                         EarlyEst.PosSigmaXZ EarlyEst.PosSigmaYZ EarlyEst.PosSigmaZZ];
    mu =     np.array([ee['lon'], ee['lat'], ee['depth']*1000.0])
    sigma =  np.array([[ee['cov_matrix']['XX'], ee['cov_matrix']['XY'], ee['cov_matrix']['XZ']], \
                       [ee['cov_matrix']['XY'], ee['cov_matrix']['YY'], ee['cov_matrix']['YZ']], \
                       [ee['cov_matrix']['XZ'], ee['cov_matrix']['YZ'], ee['cov_matrix']['ZZ']]])
    """

    x     = kwargs.get('x', None)
    mu    = kwargs.get('mu', None)
    sigma = kwargs.get('sigma', None)
    ee    = kwargs.get('ee', None)

    n = len(mu)

    #mu = np.reshape(mu,(3,1))
    mu = np.reshape(mu,(n,1))
    t1  = (2 * math.pi)**(-1*len(mu)/2)
    t2  = 1 / math.sqrt(np.linalg.det(sigma))
    #c1  = npm.repmat(mu, 1, np.shape(mu)[0])
    c1  = npm.repmat(mu, 1, len(x))
    c11 = (x - c1.transpose()).transpose()
    c12 = x - c1.transpose()

    d  = np.linalg.lstsq(sigma, c11, rcond=None)[0]
    e = np.dot(c12, d)
    f = np.multiply(-0.5,np.diag(e))
    g = np.exp(f)
    h = t1 * t2 * g

    return h
