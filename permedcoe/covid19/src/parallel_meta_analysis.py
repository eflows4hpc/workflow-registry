from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN
import xml
import scipy.io

@task(returns=2, final_xml=FILE_IN, final_cells_physicell_mat=FILE_IN)
def extract_simulation_results(final_xml, final_cells_physicell_mat):
    final_content = xml.dom.minidom.parse(final_xml)
    data_content = scipy.io.loadmat(final_cells_physicell_mat)
    # HERE WE CAN CONVERT THE CONTENTS TO A DS-ARRAY TO BE USED LATER WITH
    # A DISLIB ALGORITHM
    return final_content, data_content


def merge_reduce(function, data, chunk=48):
    """ Apply function cumulatively to the items of data,
        from left to right in binary tree structure, so as to
        reduce the data to a single value.
    :param function: function to apply to reduce data
    :param data: List of items to be reduced
    :param chunk: Number of elements per reduction
    :return: result of reduce the data to a single value
    """
    while(len(data)) > 1:
        dataToReduce = data[:chunk]
        data = data[chunk:]
        data.append(function(*dataToReduce))
    return data[0]

@task(returns=dict)
def reduce_simulation_results(*data):
    reduce_value = data[0]
    for i in range(1, len(data)):
        a, b = reduce_simulation(reduce_value, data[i])
        reduce_value.append(a)
        reduce_value.append(b)
    return reduce_value


@task(returns=dict)
def reduce_patient_results(*data):
    reduce_value = data[0]
    for i in range(1, len(data)):
        a, b = reduce_simulation(reduce_value, data[i])
        reduce_value.append(a)
        reduce_value.append(b)
    return reduce_value


def reduce_simulation(a, b):
    """
    Reduce method to accumulate two simulations
    :param a: Simulation A
    :param b: Simulation B
    :return: (Simulation A, Simulation B)
    """
    # This method should extract the necessary parameters
    return a, b
