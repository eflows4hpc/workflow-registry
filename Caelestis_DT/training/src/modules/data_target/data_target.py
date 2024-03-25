from abc import ABC, abstractmethod


class DataTarget(ABC):

    def __init__(self, route):
        self.route = route

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def set_data(self, data):
        pass

    @abstractmethod
    def read(self, source):
        pass

    @abstractmethod
    def write(self, source, data):
        pass

    def clear_data(self):
        pass
