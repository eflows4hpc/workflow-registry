from pycompss.api.task import task
from pycompss.runtime.management.classes import Future
from pycompss.api.api import compss_wait_on

@task(returns=list)
def distribute_tasks(simulation_workflow, parameters, execution_number, **simulation_parameters):
    return simulation_workflow(parameters, execution_number, **simulation_parameters)

class Simulation:
    def __init__(self, simulator_core):
        if simulator_core is not None and callable(simulator_core):
            self.simulator_core = simulator_core
        elif simulator_core is None:
            self.simulator_core = None
        else:
            raise ValueError("Simulator Engine should be callable")
        self.simulation_parameters = None
        self.simulation_generated_data = None
        self.is_workflow = True

    def set_simulator_core(self, simulator_core):
        """

        :param simulator_core:
        :return:
        """
        if simulator_core is not None and callable(simulator_core):
            self.simulator_core = simulator_core
        elif simulator_core is None:
            self.simulator_core = None
        else:
            raise ValueError("Simulator Engine should be callable")

    def set_parameters(self, parameters):
        """

        :param parameters:
        :return:
        """
        if isinstance(parameters, dict):
            self.simulation_parameters = parameters
        elif isinstance(parameters, Future):
            self.simulation_parameters = parameters
        else:
            raise ValueError("Parameters of the simulation should be sent"
                             "as a dict")

    def is_workflow(self, is_workflow=True):
        """

        :param is_workflow:
        :return:
        """
        self.is_workflow = is_workflow

    def execute(self, parameters, clear_previous_data=True):
        """

        :param parameters:
        :param clear_previous_data:
        :return:
        """
        if self.simulation_generated_data is None or clear_previous_data:
            self.simulation_generated_data = []
        if isinstance(self.simulation_parameters, Future):
            self.simulation_parameters = compss_wait_on(self.simulation_parameters)

        for exec_n, params in enumerate(parameters):
            if self.is_workflow:
                self.simulation_generated_data.append(
                    self.simulator_core(params,
                                        execution_number=exec_n,
                                        **self.simulation_parameters))
            else:
                self.simulation_generated_data.append(
                    distribute_tasks(
                        self.simulator_core, params, exec_n,
                        **self.simulation_parameters))
        return self.simulation_generated_data

    def get_simulation_data(self):
        return self.simulation_generated_data
