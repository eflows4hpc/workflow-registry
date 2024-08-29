from sampling import sampling_distribution
from simulation import prepare_alya, alya_simulation, collect_alya_results
from ai import model_selection
from pycompss.api.api import compss_wait_on
import os
import yaml
import sys
import numpy as np

def workflow(yaml, execution_dir):
    simulation_family= yaml.get("name")
    sampling_parameters = yaml.get("sampling")
    simulations_dir = os.path.join(execution_dir, "simulations")
    results_dir = os.path.join(execution_dir, "results")
    sample_set = sampling_distribution(sampling_parameters)
    sample_set = compss_wait_on(sample_set)
    simulation_parameters = yaml.get("simulation")
    y = []
    for i in range(sample_set.shape[0]):
        values = sample_set[i, :]
        name_sim = simulation_family + "-s" + str(i)
        simulation_wdir = os.path.join(simulations_dir, name_sim)
        prepare_alya(values, sampling_parameters, simulation_parameters, simulation_wdir, simulation_family, name_sim)
        alya_simulation(simulation_wdir, name_sim)
        y.append(collect_alya_results(simulation_wdir, name_sim))
    y = np.array(compss_wait_on(y))
    model_parameters = yaml.get("model_selection")
    model_selection(y, sample_set, model_parameters, results_dir) 



if __name__ == '__main__':
    yaml_file = sys.argv[1]
    execution_dir = sys.argv[2]
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    workflow(config, execution_dir)
