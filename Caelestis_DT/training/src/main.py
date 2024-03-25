from modules.auxiliar_functions_main_dt import simulation_workflow, generator_arguments
from modules.pyDOE_file import sampling, parser
from modules.data_manager.base import DataManager
from modules.data_target import Disk
from modules.digital_twin.base import DigitalTwin
from modules.simulation_phase.sampling_parameters.generator_simulation_params import GeneratorParametersSimulation
from modules.simulation_phase.simulation.simulation import Simulation
from modules.training.model_selection.base import ModelSelection
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


def main():
    #Disk and Data Manager Reads input Data
    disk = Disk("/home/bsc19/bsc19756/My_DT_Caelestis/DigitalTwin", read_file="testFernando_CIC.yaml", write_file="x_train.npy")
    dm = DataManager(disk)
    #Sampling of parameters, variables and arguments for the execution in the simulation
    gps = GeneratorParametersSimulation(sampling_generator=sampling,
                                        arguments_generator=generator_arguments,
                                        output_parser=parser)
    #Simulation
    sim = Simulation(simulation_workflow)
    #Model Selection
    model_sel = ModelSelection(scoring=r2_score)
    model_sel.set_models([DecisionTreeRegressor()])
    model_sel.set_paramaters_models([{"criterion": ["squared_error", "friedman_mse"], "max_depth": [2, 5, 10, 15, None], "random_state": [0]}])
    #Digital Twin Objetc
    dt = DigitalTwin(simulator=sim, generator_parameters_simulation=gps, data_manager=dm, model_selection=model_sel)
    best_model_instance = dt.execute()
    return best_model_instance


if __name__ == '__main__':
    main()
