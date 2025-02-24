from abc import *

from torch import nn


class BasePolicy(nn.Module):
    def __init__(self):
        super(BasePolicy, self).__init__()

    @abstractmethod
    def set_policy_id(self):
        pass

    @abstractmethod
    def zero_init(self):
        pass

    @abstractmethod
    def uniform_init(self):
        pass

    @abstractmethod
    def xavier_init(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_param_list(self):
        pass

    @abstractmethod
    def set_param_list(self):
        pass
