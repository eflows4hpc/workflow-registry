# -------------------------------------------------------------------------------------------------------------------- #
# insitu_strengths.py
# PhD thesis - Gerard Guillamet  <gerard.guillamet@udg.edu>
# Supervised by Albert Turon <albert.turon@udg.edu>
# Date: 07 April 2015
# -------------------------------------------------------------------------------------------------------------------- #
# References:
#
# Camanho, P.P, Davila, C.G., Pinho, S.T., Iannucci, L., Robinson, P. Prediction of in situ strengths and matrix
# cracking in composites under transverse tension and in-plane shear. Composites Part A: Applied Science and
# Manufacturing, 2006
#
# Maimi, P., Gonzalez, E.V. and Camanho, P.P. Comment to the paper "Analysis of Progressive matrix cracking in
# composite laminates II. First ply failure' by George J. Dvorak and Norman Laws, 2013
#
# Catalanotti, G., Camanho, P.P.Marques, A.T. Three-dimensional failure criteria for fiber-reinforced laminates.
# Composite Structures, 2013
#
# Soto, A., Gonzalez, E., Maimi, P., Mayugo, J.A., Pasquali, P.R., Camanho, P.P. A methodology to simulate
# low velocity impact and compression after impact in large composite stiffened panels. Composite Structres, 2018
#
# Furtado, G., Catalanotti, G., Arteiro, A., Gray, P.J., Wardle, B.L, CAmanho, P.P. Simulation of failure in
# laminated polymer composites: Building-block validation, Composite Structures, 2019
#
#
from math import atan, sin, cos, sqrt, pi, tan, acos, atan2
from numpy import dot, transpose, zeros, linalg, array, concatenate, seterr, trace, identity, outer

def in_situ_IN(t, e11, e22, nu12, g12, yt, yc, sl, Kplas, Splas, Gsl, g_ic, **kwargs):
    """
    In-situ strenghts function for CAELESTIS project
    """
    ao22 = 2.*(1/e22 - nu12**2/e11)              # Corrected
    # ao22 = 2.*(1/e22 - (nu12*e22/e11)**2/e11)  # Not corrected

    # Tensile transverse in-situ strengths
    yt_is1 = sqrt(8.*g_ic/pi/t/ao22)         # For a thin embedded ply
    yt_is3 = 1.12*sqrt(2.)*yt                # For a thick ply


    # In-situ matrix tension (Camanho et al. 2006)
    yt_is = max(yt_is1, yt_is3, yt)
    # In-situ shear strength including plasticity (Soto et al. 2018)
    sl_is_thick = sqrt(  (Kplas + 1.)*(2.*sl**2*Kplas + 2.*sl**2 - Splas**2) ) / (Kplas + 1.)
    sl_is_thin  = sqrt(t*(Kplas + 1.)*(pi*(Splas**2)*t + 8.*g12*Gsl*Kplas) ) / ( (Kplas + 1.)*t*sqrt(pi) )
    sl_is = max(sl_is_thick, sl_is_thin, sl)
    # In-situ matrix compression strength (Furtado et al. 2019)
    yc_is = sl_is*yc/sl
    
    return yt_is, yc_is, sl_is


def in_situ_OUT(t, e11, e22, nu12, g12, yt, yc, sl, Kplas, Splas, Gsl, g_ic, **kwargs):
    """
    In-situ strenghts function for CAELESTIS project
    """
    ao22 = 2. * (1 / e22 - nu12 ** 2 / e11)  # Corrected
    # ao22 = 2.*(1/e22 - (nu12*e22/e11)**2/e11)  # Not corrected

    # Tensile transverse in-situ strengths
    yt_is2 = 1.79 * sqrt(g_ic / pi / t / ao22)  # For a thin outer ply

    # In-situ matrix tension (Camanho et al. 2006)
    yt_is = max(yt_is2, yt)
    sl_is_out = sqrt(t * (Kplas + 1.) * (pi * (Splas ** 2) * t + 4. * g12 * Gsl * Kplas)) / (
                (Kplas + 1.) * t * sqrt(pi))
    # In-situ shear strength including plasticity (Soto et al. 2018)
    sl_is = max(sl_is_out, sl)
    # In-situ matrix compression strength (Furtado et al. 2019)
    yc_is = sl_is * yc / sl

    return yt_is, yc_is, sl_is
