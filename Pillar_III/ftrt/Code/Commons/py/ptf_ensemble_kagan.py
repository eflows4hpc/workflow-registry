import numpy as np
from scipy.stats import expon
from copy import deepcopy

def compute_ensemble_kagan(**kwargs):

    Config         = kwargs.get('cfg', None)
    ee             = kwargs.get('event_parameters', None)
    args           = kwargs.get('args', None)
    ptf_out        = kwargs.get('ptf_out', None)
    focal_mech     = kwargs.get('focal_mechanism', None)
    type_ens       = kwargs.get('type_ens', None)

    NbrFM = int(Config.get('Sampling','NbrFM'))

    if type_ens == 'RS':

       ensemble=ptf_out['new_ensemble_'+type_ens]
       Nsize=len(ensemble)
       if NbrFM>1:
          focal_mech=focal_mech[0]
          Nfm=len(focal_mech)
       else:
          Nfm=1
          focal_mech=focal_mech[0][0]
       
       for Nid in range(Nsize):

          size=len(ensemble[Nid]['real_par_scenarios_bs'][:,0])
          ptf_out['new_ensemble_'+type_ens][Nid]['real_kagan_angle']=np.zeros(size)
          ptf_out['new_ensemble_'+type_ens][Nid]['real_kagan_proba']=np.zeros((Nfm,size))
          kagan=np.zeros((Nfm,size))
          kagan_weight=np.zeros((Nfm,size))
          kagan_proball_temp=ptf_out['new_ensemble_'+type_ens][Nid]['RealProbScenBS']
 
          for iscenbs in range(size):
              
              kag_weights=np.zeros(len(focal_mech))

              for k in range(Nfm):

                 if NbrFM>1:
                    strike_fm=focal_mech[k,0]
                    dip_fm=focal_mech[k,1]
                    rake_fm=focal_mech[k,2]
                 else:   
                    strike_fm=focal_mech[0]
                    dip_fm=focal_mech[1]
                    rake_fm=focal_mech[2]

                 strike=ensemble[Nid]['real_par_scenarios_bs'][iscenbs,5]
                 dip=ensemble[Nid]['real_par_scenarios_bs'][iscenbs,6]
                 rake=ensemble[Nid]['real_par_scenarios_bs'][iscenbs,7]
                 kagan[k,iscenbs] = get_kagan_angle(strike_fm,dip_fm,rake_fm,strike,dip,rake)

          ptf_out['new_ensemble_'+type_ens][Nid]['real_kagan_angle']=kagan
          for k in range(Nfm):
              #kagan_weight[k]=vonmises.pdf(kagan[k],30.0,loc=0,scale=1)
              kagan_weight[k]=expon.pdf(kagan[k],loc=0,scale=100)
              kagan_weight[k]=kagan_weight[k]/np.sum(kagan_weight[k,:])
              kagan_proball_temp=np.multiply(kagan_proball_temp[:],kagan_weight[k,:])

          ptf_out['new_ensemble_'+type_ens][Nid]['real_kagan_proba']=kagan_proball_temp/np.sum(kagan_proball_temp)
          ptf_out['new_ensemble_'+type_ens][Nid]['RealProbScenBS']=kagan_proball_temp/np.sum(kagan_proball_temp)

    return ptf_out

def get_kagan_angle(strike1, dip1, rake1, strike2, dip2, rake2):
    """Calculate the Kagan angle between two moment tensors defined by strike,dip and
    rake.
    Kagan, Y. "Simplified algorithms for calculating double-couple rotation",
    Geophysical Journal, Volume 171, Issue 1, pp. 411-418.
    Args:
        strike1 (float): strike of slab or moment tensor
        dip1 (float): dip of slab or moment tensor
        rake1 (float): rake of slab or moment tensor
        strike2 (float): strike of slab or moment tensor
        dip2 (float): dip of slab or moment tensor
        rake2 (float): rake of slab or moment tensor
    Returns:
        float: Kagan angle between two moment tensors
    """
    # convert from strike, dip , rake to moment tensor
    tensor1 = plane_to_tensor(strike1, dip1, rake1)
    tensor2 = plane_to_tensor(strike2, dip2, rake2)

    kagan = calc_theta(tensor1, tensor2)

    return kagan

def calc_theta(vm1, vm2):
    """Calculate angle between two moment tensor matrices.
    Args:
        vm1 (ndarray): Moment Tensor matrix (see plane_to_tensor).
        vm2 (ndarray): Moment Tensor matrix (see plane_to_tensor).
    Returns:
        float: Kagan angle (degrees) between input moment tensors.
    """
    # calculate the eigenvectors of either moment tensor
    V1 = calc_eigenvec(vm1)
    V2 = calc_eigenvec(vm2)

    # find angle between rakes
    th = ang_from_R1R2(V1, V2)

    # calculate kagan angle and return
    for j in range(3):
        k = (j + 1) % 3
        V3 = deepcopy(V2)
        V3[:, j] = -V3[:, j]
        V3[:, k] = -V3[:, k]
        x = ang_from_R1R2(V1, V3)
        if x < th:
            th = x
    return th * 180.0 / np.pi


def calc_eigenvec(TM):
    """Calculate eigenvector of moment tensor matrix.
    Args:
        ndarray: moment tensor matrix (see plane_to_tensor)
    Returns:
        ndarray: eigenvector representation of input moment tensor.
    """

    # calculate eigenvector
    V, S = np.linalg.eigh(TM)
    inds = np.argsort(V)
    S = S[:, inds]
    S[:, 2] = np.cross(S[:, 0], S[:, 1])
    return S

def ang_from_R1R2(R1, R2):
    """Calculate angle between two eigenvectors.
    Args:
        R1 (ndarray): eigenvector of first moment tensor
        R2 (ndarray): eigenvector of second moment tensor
    Returns:
        float: angle between eigenvectors
    """

    #    return np.arccos((np.trace(np.dot(R1, R2.transpose())) - 1.) / 2.)
    return np.arccos(np.clip((np.trace(np.dot(R1, R2.transpose())) - 1.0) / 2.0, -1, 1))

def plane_to_tensor(strike, dip, rake, mag=6.0):
    """Convert strike,dip,rake values to moment tensor parameters.
    Args:
        strike (float): Strike from (assumed) first nodal plane (degrees).
        dip (float): Dip from (assumed) first nodal plane (degrees).
        rake (float): Rake from (assumed) first nodal plane (degrees).
        magnitude (float): Magnitude for moment tensor
            (not required if using moment tensor for angular comparisons.)
    Returns:
        nparray: Tensor representation as 3x3 numpy matrix:
            [[mrr, mrt, mrp]
            [mrt, mtt, mtp]
            [mrp, mtp, mpp]]
    """
    # define degree-radian conversions
    d2r = np.pi / 180.0
    # get exponent and moment magnitude
    magpow = mag * 1.5 + 16.1
    mom = np.power(10, magpow)

    # get tensor components
    mrr = mom * np.sin(2 * dip * d2r) * np.sin(rake * d2r)
    mtt = -mom * ((np.sin(dip * d2r) * np.cos(rake * d2r) * np.sin(2 * strike * d2r)) +
                  (np.sin(2 * dip * d2r) * np.sin(rake * d2r) *
                  (np.sin(strike * d2r) * np.sin(strike * d2r))))
    mpp = mom * ((np.sin(dip * d2r) * np.cos(rake * d2r) * np.sin(2 * strike * d2r)) -
                 (np.sin(2 * dip * d2r) * np.sin(rake * d2r) *
                 (np.cos(strike * d2r) * np.cos(strike * d2r))))
    mrt = -mom * ((np.cos(dip * d2r) * np.cos(rake * d2r) * np.cos(strike * d2r)) +
                  (np.cos(2 * dip * d2r) * np.sin(rake * d2r) * np.sin(strike * d2r)))
    mrp = mom * ((np.cos(dip * d2r) * np.cos(rake * d2r) * np.sin(strike * d2r)) -
                 (np.cos(2 * dip * d2r) * np.sin(rake * d2r) * np.cos(strike * d2r)))
    mtp = -mom * ((np.sin(dip * d2r) * np.cos(rake * d2r) * np.cos(2 * strike * d2r)) +
                  (0.5 * np.sin(2 * dip * d2r) * np.sin(rake * d2r) *
                  np.sin(2 * strike * d2r)))

    mt_matrix = np.array([[mrr, mrt, mrp],
                          [mrt, mtt, mtp],
                          [mrp, mtp, mpp]])
    mt_matrix = mt_matrix * 1e-7  # convert from dyne-cm to N-m
    return mt_matrix

def find_nearest(array, value):
    arr = np.asarray(array)
    idx = 0
    diff = arr-value
    diff[diff<1e-26]=100.0
    idx=diff.argmin()
    return idx,array[idx]

