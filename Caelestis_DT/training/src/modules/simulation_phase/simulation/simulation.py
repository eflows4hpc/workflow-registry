from pycompss.api.task import task
from pycompss.runtime.management.classes import Future
from pycompss.api.api import compss_wait_on

@task(returns=list)
def distribute_tasks(simulation_workflow, parameters, execution_number, **simulation_parameters):
    return simulation_workflow(parameters, execution_number, **simulation_parameters)

class Simulation:
    """
    Object in charge of executing the simulation workflow or to distribute the different calls to the simulator core
    """
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
        Sets the simulator to use to generate the training data.
        :param simulator_core: function that implements the workflow of the simulation
        or the call to the simulation.
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
        Set the parameters that will be used in the different simulations
        :param parameters: dict or list, containing sets of parameters
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
        Specify if the simulator core is a workflow or
        just a call to the simulator engine
        :param is_workflow: boolean
        :return:
        """
        self.is_workflow = is_workflow

    def execute(self, parameters, clear_previous_data=True):
        """
        Executes the workflow of the call to the simulator engine
        :param parameters: list containing the parameters for each simulation
        :param clear_previous_data: boolean,
        specify to clean the data already generated or accumulate it
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
        """
        Returns the data generated in the execution of the simulation
        """
        return self.simulation_generated_data
