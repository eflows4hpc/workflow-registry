from pyDOE import *
from scipy.stats.distributions import norm, uniform, weibull_min, lognorm
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import *

@task(returns=1)
def sampling_distribution(problem):
    variables = problem.get("variables-sampler")
    n_samples = int(problem.get("n_samples"))
    num_var = int(problem.get("num_var"))
    
    covs = []
    means = []
    sigmas = []
    alphas = []
    betas = [] 
    typeDistr = []
    for item in variables:
        for key, value in item.items():
            distr = value.get('distr')
            mean = float(value.get("mean"))
            sigma = float(value.get('sigma'))
            cov = float(value.get('cov'))
            alpha = float(value.get('alpha'))
            beta = float(value.get('beta'))
            typeDistr.append(distr)
            means.append(mean)
            covs.append(cov)
            sigmas.append(sigma)
            alphas.append(alpha)
            betas.append(beta)
            
    lhs_sample = lhs(num_var, n_samples, criterion="maximin")
    
    samples_final = np.zeros((n_samples, num_var))
    
    for i in range(num_var):
        if typeDistr[i] == 'norm':
            samples_final[:, i] = norm(loc=means[i], scale=sigmas[i]).ppf(lhs_sample[:, i])
        elif typeDistr[i] == 'uniform':
            lower_bound = means[i] - means[i]*covs[i]/100.
            upper_bound = means[i] + means[i]*covs[i]/100.
            samples_final[:, i] = uniform(loc=lower_bound, scale=(upper_bound - lower_bound)).ppf(lhs_sample[:, i])
        elif typeDistr[i] == 'weibull': 
            samples_final[:, i] = weibull_min(betas[i], scale=alphas[i], loc=0).ppf(lhs_sample[:, i])
        elif typeDistr[i] == 'lognorm':
            samples_final[:, i] = lognorm(s=sigmas[i], loc=means[i]).ppf(lhs_sample[:, i])
            
    return samples_final
