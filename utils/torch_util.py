import torch
import numpy as np


def get_flatten_params(model):
    """
    flattens all parameters into a single column vector. Returns the dictionary to recover them
    :param: model: a neural network pytorch instance
    :return: a dictionary: {"params": [#params,
    "indices": [(start index, end index) for each param] **Note end index in uninclusive**
    """

    param_list = model.get_param_list()
    l = [np.ravel(p) for p in param_list]
    lengths = []
    s = 0
    for p in l:
        size = p.shape[0]
        lengths.append((s, s + size))
        s += size
    flat = np.concatenate(l)
    return {"params": flat, "lengths": lengths}


def set_flatten_params(flat_params, lengths, model):
    """
    Gives a list of recovered parameters from their flattened form
    :param flat_params: [#params, 1]
    :param indices: a list detaling the start and end index of each param [(start, end) for param]
    :param model: the model that gives the params with correct shapes
    """
    param_lst = []
    flat_params_lst = [flat_params[s:e] for (s, e) in lengths]
    for param, flat_param in zip(model.get_param_list(), flat_params_lst):
        param_lst.append(np.copy(flat_param.reshape(param.shape)))
    set_param_list(model, param_lst)


def get_param_list(model):
    """
    get a list of parameters of the model
    :param model: a neural network pytorch instance
    :return: [#params for every model module]
    """
    param_lst = []
    for param in model.parameters():
        param_lst.append(param.data.numpy())
    return param_lst


def set_param_list(model, param_lst: list):
    """
    set model parameters from a list
    :param model: a neural network pytorch instance
    :param param_lst:  a list: [#params for every model module]
    """
    lst_idx = 0
    for param in model.parameters():
        param.data = torch.tensor(param_lst[lst_idx]).float()
        lst_idx += 1


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


@torch.no_grad()
def xavier_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.fill_(0.0)
