from abc import ABC, abstractmethod


class MakeTraining(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def add_validation_metric(self):
        pass

    @abstractmethod
    def print_validation_statistics(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def cross_validation(self):
        pass
