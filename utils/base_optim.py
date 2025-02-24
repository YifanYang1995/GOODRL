from abc import *


class BaseOptim(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args):
        pass

    @abstractmethod
    def init_population(self, *args):
        pass

    @abstractmethod
    def next_population(self, *args):
        pass
