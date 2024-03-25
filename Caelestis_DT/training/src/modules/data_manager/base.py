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
    def __init__(self, data_target):
        if isinstance(data_target, DataTarget):
            self.data_target = data_target
        else:
            raise ValueError("Data source should be an instance of a class"
                             "that inherits from DataSource")

    def set_data(self, data):
        _set_data(self.data_target, data)

    def get_data(self):
        return _get_data(self.data_target)

    @constraint(computing_units="${ComputingUnits}")
    @task(source=FILE_OUT)
    def read(self, source):
        return self.data_target.read(source)

    @constraint(computing_units="${ComputingUnits}")
    @task(source=FILE_IN)
    def write(self, data, source=None):
        self.data_target.write(source, data)

    def clear_data_source(self):
        self.data_target.clear_data()
