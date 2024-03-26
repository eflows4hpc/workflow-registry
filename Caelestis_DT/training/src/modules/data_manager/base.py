from modules.data_target.data_target import DataTarget
from pycompss.api.task import task
from pycompss.api.constraint import constraint
from pycompss.api.parameter import (FILE_IN, FILE_OUT, INOUT,
                                    COLLECTION_IN)


@constraint(computing_units="${ComputingUnits}")
@task(data_source=INOUT)
def _set_data(data_target, data):
    data_target.set_data(data)


@constraint(computing_units="${ComputingUnits}")
@task(returns=1)
def _get_data(data_target):
    return data_target.get_data()


class DataManager:
    """
    Object that manages the writing and reading of data from different sources.

    Parameters
    ----------
    data_target: DataTarget
        Object to read and write data from. This object can be a Disk, DataBase or StreamEngine
    """
    def __init__(self, data_target):
        if isinstance(data_target, DataTarget):
            self.data_target = data_target
        else:
            raise ValueError("Data source should be an instance of a class"
                             "that inherits from DataSource")

    def set_data(self, data):
        """
        Writes data using the Data Target
        """
        _set_data(self.data_target, data)

    def get_data(self):
        """
        Reads data using the Data Target
        """
        return _get_data(self.data_target)

    @constraint(computing_units="${ComputingUnits}")
    @task(source=FILE_OUT)
    def read(self, source):
        """
        Sets file to read data from and reads the data
        source: string, name of the file to read the data from.
        """
        return self.data_target.read(source)

    @constraint(computing_units="${ComputingUnits}")
    @task(source=FILE_IN)
    def write(self, data, source=None):
        """
        Sets file to write data into and writes the data
        data: data to write
        source: string, name of the file to write the data into.
        """
        self.data_target.write(source, data)

    def clear_data_source(self):
        """
        Deletes the data written.
        """
        self.data_target.clear_data()
