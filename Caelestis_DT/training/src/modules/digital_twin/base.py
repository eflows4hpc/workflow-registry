from pycompss.api.api import compss_wait_on, compss_barrier
from modules.simulation_phase.sampling_parameters.generator_simulation_params import GeneratorParametersSimulation
from pycompss.runtime.management.classes import Future
from dislib.data.array import Array
import numpy as np
import dislib as ds
import math
import time

from modules.training.model_selection.base import ModelSelection


class DigitalTwin:
    def __init__(self, simulator=None, data_preprocessing=None, generator_parameters_simulation=None, data_manager=None, training=None, block_size=None):
        self.simulator = simulator
        self.generator_parameters_simulation = generator_parameters_simulation
        self.data_manager = data_manager
        self.training = training
        self.data_preprocessing = data_preprocessing
        self.x_data = None
        self.y_data = None
        self.data_parsed = None
        self.arguments_simulation = None
        self.best_model = None
        self.block_size = block_size

    def set_training(self, training):
        self.training = training

    def set_simulator(self, simulator):
        self.simulator = simulator

    def set_generator_parameters_simulation(self, generator_parameters_simulation=None):
        if isinstance(generator_parameters_simulation, GeneratorParametersSimulation):
            self.generator_parameters_simulation = generator_parameters_simulation
        else:
            raise ValueError("Wrong generator parameters simulation type")

    def set_block_size(self, block_size):
        self.block_size = block_size

    def set_x_data(self, x_data):
        self.x_data = x_data

    def set_data(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def save_data_generated(self, path):
        self.path = path
        if self.x_data is not None:
            self.data_manager.set_data(self.x_data)
        if self.y_data is not None:
            self.data_manager.set_data(self.y_data)

    def visualize_results_training(self):
        self.training.visualize_results()

    def execute(self):
        configuration_data = self.data_manager.get_data()
        self._execute_generator_parameters_simulation(configuration_data)
        self._execute_simulation(self.arguments_simulation, self.data_parsed)
        x_train, y_train = self._execute_preprocessing()
        self.best_model = self._execute_training(x_train, y_train)
        return self.best_model

    def _execute_generator_parameters_simulation(self, configuration_data):
        if self.generator_parameters_simulation is not None:
            self.generator_parameters_simulation.set_parameters(configuration_data)
            self.generator_parameters_simulation.generate_sampling_simulation()
            self.data_parsed = self.generator_parameters_simulation.get_output_parsed(configuration_data)
            self.x_data = self.generator_parameters_simulation.get_data_generated()
            self.arguments_simulation = self.generator_parameters_simulation.generate_arguments_simulation(arguments_key="input",
                                                                                                           data_folder="/home/bsc19/bsc19756/Caelestis_Data/")
        else:
            return

    def _execute_simulation(self, arguments, data):
        if self.simulator is not None:
            self.simulator.set_parameters(arguments)
            self.y_data = self.simulator.execute(data)
        else:
            return

    def _execute_preprocessing(self):
        if isinstance(self.y_data, Future):
            self.y_data = np.array(compss_wait_on(self.y_data))
            if self.y_data.ndim == 1:
                self.y_data = self.y_data[:, np.newaxis]
        elif isinstance(self.y_data, list):
            if isinstance(self.y_data[0], Future):
                self.y_data = np.array(compss_wait_on(self.y_data))
                if self.y_data.ndim == 1:
                    self.y_data = self.y_data[:, np.newaxis]
        elif self.y_data is None:
            raise ValueError("Not specified Y to train, neither Simulation")
        if self.x_data is None:
            raise ValueError("Not specified training data neither generator")
        if isinstance(self.x_data, Future):
            self.x_data = np.array(compss_wait_on(self.x_data))
        elif isinstance(self.x_data, list):
            if isinstance(self.x_data[0], Future):
                self.x_data = np.array(compss_wait_on(self.x_data))
        if isinstance(self.y_data, np.ndarray) and self.y_data.ndim == 1:
            self.y_data = self.y_data[:, np.newaxis]
        if self.block_size is None:
            self.block_size = (math.ceil(self.x_data.shape[0] / 2), self.x_data.shape[1])
        if isinstance(self.x_data, Array):
            x_train = self.x_data
        else:
            x_train = ds.array(self.x_data, block_size=self.block_size)
        if isinstance(self.y_data, Array):
            y_train = self.y_data
        else:
            y_train = ds.array(self.y_data, block_size=(self.block_size[0], 1))
        if self.data_preprocessing is not None:
            self.data_preprocessing.fit_transform(self.x_data, self.y_data)
        return x_train, y_train

    def _execute_training(self, x_train, y_train):
        if isinstance(self.training, ModelSelection):
            self.training.execute_grid_search(x_train, y_train)
        self.best_model = self.training.fit(x_train, y_train)
        if self.save_model:
            self.training.save_model()
        return self.best_model

