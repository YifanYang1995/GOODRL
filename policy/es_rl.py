from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.base_optim import BaseOptim
# from utils.optimizers import Adam
from utils.policy_dict import agent_policy
from utils.torch_util import get_flatten_params, set_flatten_params
from collections import deque
from utils.base_model import BasePolicy
from config.Params import configs
from policy.actor3 import Actor
device = torch.device(configs.device)

class ESOpenAI(BaseOptim):
    def __init__(self, policy_name,
                 input_dim_wf,
                 input_dim_vm,
                 hidden_dim,
                 gnn_layers,
                 mlp_layers,
                 para = 'None',
                 heads=1,
                 dropout=0.0,
                 activate_fn = 'relu',
                 embedding_type = 'gat',):
        super(ESOpenAI, self).__init__()
        # self.name = config["name"]
        self.sigma_init = configs.sigma_init  # noise standard deviation
        self.sigma_curr = self.sigma_init
        self.sigma_decay = configs.sigma_decay
        self.learning_rate = configs.lr
        self.population_size = configs.es_pop_size
        self.reward_norm = configs.normalize_rewards

        self.epsilons = []  # save epsilons with respect to every model

        self.agent_ids = None
        # self.policy = None
        # self.policy_flatten_params = None  
        # self.optimizer = None

        if policy_name == 'gat':
            self.policy = Actor(input_dim_wf,
                                input_dim_vm,           
                                hidden_dim,
                                gnn_layers,
                                mlp_layers,                                                            
                                # heads,
                                # dropout,     
                                # activate_fn,   
                                ).to(device)
        else:
            self.policy = WFPolicy(policy_name)
        self.policy.norm_init()

        if para is not None:
            self.policy.load_state_dict(para)      

        self.policy_flatten_params = torch.tensor(get_flatten_params(self.policy)['params'],
                                                    requires_grad=True, dtype=torch.float64)
        self.optimizer = torch.optim.Adam([self.policy_flatten_params], lr=self.learning_rate)


    # Init policies of θ_t and (θ_t + σϵ_i)
    def init_population(self):
        # first, init θ_t
        # self.agent_ids = 
        # policy.norm_init()
        # self.policy = policy
        # self.policy_flatten_params = torch.tensor(get_flatten_params(self.policy)['params'],
        #                                             requires_grad=True, dtype=torch.float64)
        # self.optimizer = torch.optim.Adam([self.policy_flatten_params], lr=self.learning_rate)

        # second, init (θ_t + σϵ_i)
        perturbations = self.init_perturbations(self.policy, self.sigma_curr, self.population_size)
        return perturbations

    def init_perturbations(self, mu_model: torch.nn.Module, sigma, pop_size):
        perturbations = []  # policy F_i
        self.epsilons = []  # epsilons list

        # add mu model to perturbations for future evaluation
        perturbations.append(deepcopy(mu_model))

        # init eps as 0 (a trick for the implementation only)
        zero_eps = deepcopy(mu_model)
        zero_eps.zero_init()
        zero_eps_param_lst = get_flatten_params(zero_eps)
        self.epsilons.append(zero_eps_param_lst['params'])

        # a loop of producing perturbed policy
        for _num in range(pop_size):
            perturbed_policy = deepcopy(mu_model)
            perturbed_policy.set_policy_id(_num)

            perturbed_policy_param_lst = get_flatten_params(perturbed_policy)  # θ_t
            epsilon = np.random.normal(size=perturbed_policy_param_lst['params'].shape)  # ϵ_i
            perturbed_policy_param_updated = perturbed_policy_param_lst['params'] + epsilon * sigma  # θ_t + σϵ_i

            set_flatten_params(perturbed_policy_param_updated, perturbed_policy_param_lst['lengths'], perturbed_policy)

            perturbations.append(deepcopy(perturbed_policy))
            self.epsilons.append(epsilon)  # append epsilon for current generation

        return perturbations

    def next_population(self, rewards):
        # rewards = results['rewards'].tolist()
        # best_reward_per_g = max(rewards)
        # rewards = np.array(rewards)
        rewards = -rewards

        # fitness shaping
        rewards = self.compute_centered_ranks(rewards)
        # normalization
        # if self.reward_norm:
        #     r_std = rewards.std()
        #     rewards = (rewards - rewards.mean()) / r_std

        # init next mu model
        update_factor = 1 / ((len(self.epsilons) - 1) * self.sigma_curr)  # epsilon -1 because parent policy is included
        update_factor *= -1.0  # adapt to minimization

        # sum of (F_j * epsilon_j)
        grad_param_list = np.sum(np.array(self.epsilons) * rewards.reshape(rewards.shape[0], 1), axis=0)
        grad_param_list *= update_factor
        mean_grad = np.mean(grad_param_list)

        # Update paras using Adam
        self.policy_flatten_params.grad = torch.tensor(grad_param_list, dtype=torch.float64)  # set grad to corresponding para
        self.optimizer.step()
        flatten_params = self.policy_flatten_params.clone()
        set_flatten_params(flatten_params.detach().numpy(), get_flatten_params(self.policy)['lengths'], self.policy)

        # Continue with the generation of new perturbations
        perturbations = self.init_perturbations(self.policy, self.sigma_curr, self.population_size)

        if self.sigma_curr >= 0.01:
            self.sigma_curr *= self.sigma_decay

        return perturbations, self.sigma_curr, mean_grad

    def get_elite_model(self):
        return self.policy
    
    def compute_ranks(self,x):
        """
        Returns ranks in [0, len(x))
        Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
        """
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    def compute_centered_ranks(self,x):
        y = self.compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return y


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        with torch.no_grad():  # Will not call Tensor.backward()
            x = torch.from_numpy(x).float()
            x = x.unsqueeze(0)  # Todo: check x dim as its condition
            x = torch.tanh(self.fc1(x))  # Activation function: tanh
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)

            # Apply softmax to get the probability distribution
            x = F.softmax(x.squeeze(), dim=0)  # All dimensions of input of size 1 removed
            
            # Compute the entropy using Categorical
            dist = torch.distributions.Categorical(probs=x)
            entropy = dist.entropy()

            # Select action
            a = torch.argmax(x)

            # Detach and convert to numpy
            a = a.detach().cpu().numpy()

            return a.item(0), entropy.item()

class Trans(nn.Module):
    def __init__(self,task_fea_size, vm_fea_size, output_size,
                 d_model, num_heads, num_en_layers, d_ff, dropout):
        super(Trans,self).__init__()
        # self.task_dim = task_fea_size
        self.vm_dim = vm_fea_size
        self.trans = SelfAttentionEncoder(task_fea_size, vm_fea_size, output_size,
                 d_model, num_heads, num_en_layers, d_ff, dropout)
        
    def forward(self, x):
        with (torch.no_grad()):  # Will not call Tensor.backward()
            x = torch.from_numpy(x).float()
            # print(f'x进入网络的形状{x.shape}, x进入网络的内容{x}, \n')
            task_info = x[:, 0:-self.vm_dim].unsqueeze(1)  # 序列长度=VM数, 每个序列内容为当前task特征信息(所有VM都一样)
            vm_info = x[:, -self.vm_dim::].unsqueeze(1)  # 序列长度=VM数, 每个序列内容为一个vm的特征信息
            x = self.trans(task_info, vm_info)
            # print(f'Policy输出的形状{x.shape}, Policy输出的内容{x}, \n')
            # x = x.permute(1, 0, 2)  # x输出形状改为和Alba一样的输出形式
            # x = torch.argmax(x.squeeze())
            # x = x.detach().cpu().numpy()
            # return x.item(0)
        
            x = F.softmax(x.squeeze(), dim=0)  # All dimensions of input of size 1 removed
            
            # Compute the entropy using Categorical
            dist = torch.distributions.Categorical(probs=x)
            entropy = dist.entropy()

            # Select action
            a = torch.argmax(x)

            # Detach and convert to numpy
            a = a.detach().cpu().numpy()

            return a.item(0), entropy.item()        
        
class SelfAttentionEncoder(nn.Module):
    def __init__(self, task_fea_size, vm_fea_size, output_size,
                 d_model, num_heads, num_en_layers, d_ff, dropout=0.1):
        super(SelfAttentionEncoder, self).__init__()
        # Task task_preprocess
        self.task_embedding = nn.Sequential(nn.Linear(task_fea_size, d_model))
        self.task_feature_enhance = nn.Sequential(nn.Linear(d_model, 2*d_model),
                                                  nn.ReLU(),
                                                  nn.Linear(2*d_model, d_model))

        # VM task_preprocess
        self.vm_embedding = nn.Sequential(nn.Linear(vm_fea_size, d_model))

        # self-attention
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_en_layers)

        # priority mapping
        self.priority = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model),
                                      nn.ReLU(),
                                      nn.Linear(2 * d_model, d_model),
                                      nn.ReLU(),
                                      nn.Linear(d_model, output_size))


    def forward(self, task_info, vm_info):
        # Task task_preprocess
        task_embedded = self.task_embedding(task_info)
        task_feature_enhance = self.task_feature_enhance(task_embedded)

        # VM task_preprocess
        vm_embedded = self.vm_embedding(vm_info)

        # self-attention
        # 设vm_embedded的形状为 (10, 1, 20)=(seq_length, batch_size, feature_dim) 即:vm=10, 每个vm=1x20的batch-data
        # 则进入transformer的形状应为 (1, 10, 20)=(batch_size, seq_length, feature_dim), 因为设置了batch_first=True
        vm_embedded = vm_embedded.permute(1, 0, 2)
        global_info = self.transformer_encoder(vm_embedded)  # 编码器输出学习到的全局注意信息
        global_info = global_info.permute(1, 0, 2)

        # Feature concatenation
        concatenation_features = torch.cat((global_info, task_feature_enhance), dim=-1)  # 沿最后一维即特征维拼接

        # priority mapping
        priority = self.priority(concatenation_features)

        return priority


class WFPolicy(BasePolicy):
    def __init__(self,policy_name, policy_id=-1):
        super(WFPolicy, self).__init__()
        self.policy_id = policy_id  # Parent policy when id = -1, Child policy id >= 0
        self.state_num = configs.input_dim_wf + configs.input_dim_vm + 1
        self.action_num = configs.vm_types*configs.each_vm_type_num

        if policy_name == 'WFPolicy':
            self.model = MLP(self.state_num,1)
        elif policy_name == 'Trans':
            self.model = Trans(task_fea_size=4, vm_fea_size=4, output_size=1, d_model=16,
                                          num_heads=2, num_en_layers=2, d_ff=64, dropout=0.1)       
        
    def norm_init(self, std=1.0):
        for param in self.parameters():
            shape = param.shape
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            param.data = torch.from_numpy(out)        

    def get_param_list(self):
        param_lst = []
        for param in self.parameters():
            param_lst.append(param.data.numpy())
        return param_lst            
    
    def zero_init(self):
        for param in self.parameters():
            param.data = torch.zeros(param.shape)

    def set_policy_id(self, policy_id):
        self.policy_id = policy_id            