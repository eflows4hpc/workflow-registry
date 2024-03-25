import numpy as np
from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.api import compss_wait_on
from pycompss.runtime.management.classes import Future


def gaussian_distribution(n_samples, parameters_data):
    """

    :param n_samples:
    :param parameters_data:
    :return:
    """
    mean_values = []
    cov_values = []
    fixed_values = []
    for key in parameters_data.keys():
        if isinstance(parameters_data[key], list):
            if len(parameters_data[key]) == 1:
                fixed_values.append(parameters_data[key][0])
            else:
                mean_values.append(parameters_data[key][0])
                cov_values.append(parameters_data[key][1])
        elif isinstance(parameters_data[key], dict):
            if "mean" in parameters_data[key].keys():
                mean_values.append(parameters_data[key]["mean"])
            if "cov" in parameters_data[key].keys():
                cov_values.append(parameters_data[key]["cov"])
            if "fixed_value" in parameters_data[key].keys():
                fixed_values.append(parameters_data[key]["fixed_value"])
    fixed_values = np.array([fixed_values])
    fixed_values = np.repeat(fixed_values, n_samples, axis=0)
    distribution = np.random.multivariate_normal(
        mean_values, cov_values, size=n_samples)
    distribution = np.hstack([fixed_values, distribution])
    return distribution


class GeneratorParametersSimulation:
    """
    Object that samples the parameters for the different simulations and parses
    the parameters to the format required to be used as input for the simulation

    Parameters
    ----------
    sampling_generator: callable
        Function or workflow that samples the parameters of the simulation in the whole parameter space
    arguments_generator: callable

    parameters: dict or Future
        parameters and their ranges to sample
    output_parser: callable
        Function that will parse the sampled parameters as needed for their usage in the simulation

    """
    def __init__(self, sampling_generator=None, arguments_generator=None, parameters=None,
                 distribution=None, output_parser=None, elements_to_parse=None, output_format=None):
        self.distribution_list = {"gaussian_distribution":
                                  gaussian_distribution}
        self.sampling_generator = sampling_generator
        self.arguments_generator = arguments_generator
        if isinstance(parameters, dict):
            if "problem" in parameters.keys():
                self.parameters = parameters["problem"]
            else:
                self.parameters = parameters
            if "input" in parameters.keys():
                self.parameters_argument_generator = parameters["input"]
            else:
                self.parameters_argument_generator = None
        elif isinstance(parameters, Future):
            self.parameters, self.parameters_argument_generator = \
                _set_parameters_from_future(parameters)
        if (distribution is not None and
                distribution in self.distribution_list):
            self.distribution = distribution
        else:
            self.distribution = None
        self.data_generated = None
        self.arguments_simulation = None
        self.output_parser = output_parser
        if elements_to_parse is not None:
            if isinstance(elements_to_parse, int):
                self.elements_to_parse = elements_to_parse
            else:
                self.elements_to_parse = 1
                raise Warning("Invalid format for elements_to_parse argument."
                              "Assigning 1 to elements_to_parse.")
        else:
            self.elements_to_parse = 1
        self.available_output_formats = ["list", "list_dicts", "dict"]
        if output_format != "list" and self.sampling_generator is None:
            if output_format in self.available_output_formats:
                self.output_format = output_format
            else:
                self.output_format = "list"
                raise Warning("Output format not available, "
                              "assigning the default format: list.")
        elif self.sampling_generator is not None:
            pass
        else:
            self.output_format = "list"

    def get_data_generated(self):
        """
        Function to recover the data generated from the sampling
        returns:
        data
        """
        return self.data_generated

    def set_sampling_generator(self, sampling_generator):
        """
        This function will be used to set the sampling generator
        :param sampling_generator:
        :return:
        """
        self.sampling_generator = sampling_generator

    def set_output_parser(self, output_parser):
        """
        This function will set the parser for the output data of this component
        :param output_parser:
        :return:
        """
        self.output_parser = output_parser

    def set_output_format(self, output_format):
        """

        :param output_format:
        :return:
        """
        if output_format in self.available_output_formats:
            self.output_format = output_format
        else:
            raise Warning("Output format not available, "
                          "assigning the default format: list.")

    def set_parameters(self, parameters):
        """
        This function will receive parameters and their fix value or range of values to sample.
        This parameters are going to be sampled and used as input for a simulation.

        :param parameters: dict. Should contain as keys strings with the name of the parameters.
        In the values it should contain one of the following options:
            - A list with one unique value. It will define a fixed value for that parameter.
            - A list with two values. The first value will be the mean of the sampling of that parameters,
            the second value will be the covariance of the sampling.
            - A dictionary. It should contain a key fixed_value in the case that the parameters will have
            the same value in all the samplings or a key for mean value and another cov for covariance value.
        :return: void
        """
        if isinstance(parameters, dict):
            if "problem" in parameters.keys():
                self.parameters = parameters["problem"]
            else:
                self.parameters = parameters
            if "input" in parameters.keys():
                self.parameters_argument_generator = parameters["input"]
            else:
                self.parameters_argument_generator = None
        elif isinstance(parameters, Future):
            self.parameters, self.parameters_argument_generator = \
                _set_parameters_from_future(parameters)

        else:
            raise ValueError("This function should receive a dictionary"
                             "as the input")

    def set_elements_to_parse(self, elements_to_parse):
        """

        :param elements_to_parse:
        :return:
        """
        self.elements_to_parse = elements_to_parse

    def set_distribution(self, distribution):
        """

        :param distribution:
        :return:
        """
        if isinstance(distribution, str):
            if distribution in self.distribution_list.keys():
                self.distribution = distribution
            else:
                raise ValueError("This distribution is not available")
        elif callable(distribution):
            self.distribution = distribution
        else:
            raise ValueError("Distribution is not a callable function neither"
                             "one of the available distributions.")

    def get_output_parsed(self, configuration_data):
        """
        Function that will parse the data generated by applying the parser specified or to the corresponding output format
        :param kwargs:
        :return:
        data parsed
        """
        if isinstance(self.data_generated, Future):
            self.data_generated = compss_wait_on(self.data_generated)
        data_parsed = []
        if self.output_parser is not None:
            for i in range(0, len(self.data_generated), self.elements_to_parse):
                elements_parsed = [self.output_parser(
                    self.data_generated[i: i + self.elements_to_parse],
                    configuration_data)]
                data_parsed.extend(elements_parsed)
        elif self.output_format is not None:
            if self.output_format == "list":
                if (isinstance(self.data_generated, list) and
                        isinstance(self.data_generated[0], list)):
                    return self.data_generated
                else:
                    pass
            elif self.output_format == "dict":
                if isinstance(self.data_generated, dict):
                    return self.data_generated
                else:
                    pass
            else:
                if (isinstance(self.data_generated, list) and
                        isinstance(self.data_generated[0], dict)):
                    return self.data_generated
                else:
                    pass
        else:
            raise ValueError("Output parser is not specified")
        return data_parsed

    def generate_sampling_simulation(self, n_samples=None, key="n_samples",
                                     configuration_data=None):
        """

        :param n_samples: int or dict with "n_samples" key
        :return:
        """
        if n_samples is not None:
            if isinstance(n_samples, int):
                samples_to_generate = n_samples
            elif isinstance(n_samples, dict):
                samples_to_generate = n_samples["n_samples"]
        else:
            samples_to_generate = _obtain_samples_to_generate(self.parameters, key)
        self.data_generated = None
        if self.sampling_generator is not None:
            if self.parameters is not None:
                self.data_generated = self.sampling_generator(
                    samples_to_generate,
                    self.parameters)
            else:
                pass
        elif self.distribution is not None:
            if callable(self.distribution):
                self.data_generated = self.distribution(samples_to_generate,
                                                        self.parameters)
            else:
                self.data_generated = self.distribution_list[
                    self.distribution](samples_to_generate, self.parameters)
        if self.output_parser is not None and configuration_data is not None:
            self.data_generated = self.output_parser(self.data_generated, configuration_data)
        return self.data_generated

    def generate_arguments_simulation(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        if "n_arguments" in kwargs:
            pass
        else:
            if self.parameters_argument_generator is not None:
                self.arguments_simulation = self.arguments_generator(
                    input_yaml=self.parameters_argument_generator,
                    data_folder=kwargs["data_folder"])
            elif "arguments_key" in kwargs.keys():
                self.arguments_simulation = self.arguments_generator(
                    input_yaml=self.parameters[kwargs["arguments_key"]],
                    data_folder=kwargs["data_folder"])
            else:
                self.arguments_simulation = self.arguments_generator(**kwargs)
        return self.arguments_simulation


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _obtain_samples_to_generate(parameters, key):
    if isinstance(parameters, dict):
        return parameters[key]
    else:
        return parameters


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _obtain_parameters_from_future(parameters):
    if isinstance(parameters, dict):
        if "problem" in parameters.keys():
            return parameters["problem"]
        else:
            return parameters

@constraint(computing_units="${ComputingUnits}")
@task(returns=2)
def _set_parameters_from_future(parameters):
    if isinstance(parameters, dict):
        if "problem" in parameters.keys():
            prob_parameters = parameters["problem"]
        else:
            prob_parameters = parameters
        if "input" in parameters.keys():
            parameters_argument_generator = parameters["input"]
        else:
            parameters_argument_generator = None
    return prob_parameters, parameters_argument_generator
