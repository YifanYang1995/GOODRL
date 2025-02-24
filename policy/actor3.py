import os
import sys
from copy import deepcopy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, Tanh, TransformerEncoderLayer, TransformerEncoder
from torch_geometric.nn import GINConv, GATConv, global_mean_pool
# from torch_geometric.utils import scatter, add_self_loops
from torch.distributions.categorical import Categorical
from config.Params import configs

from collections import deque

device = torch.device(configs.device)

class BatchGraph: ## wf_features, wf_edges, wf_masks, vm_features, vm_edges, vm_masks, candidate_taskID
    def __init__(self, normalize=False):
        self.candidate_taskID = None
        self.wf_features = None
        self.wf_edges = None
        self.wf_masks = None
        self.wf_batchs = None
        self.vm_features = None
        self.vm_edges = None
        self.vm_masks = None
        self.vm_batchs = None
        self.normalize = normalize

    def form_v_state(self, candidate_taskID, wf_features, wf_edges, wf_masks, wf_batchs, vm_features, vm_edges, vm_masks, vm_batchs): 

        if self.normalize:
            wf_features = self.feature_normalize1('features', wf_features)
            vm_features = self.feature_normalize1('vm', vm_features) 
 
        for i in range(vm_edges.shape[0]):
            wf_edges = np.column_stack([wf_edges, vm_edges[i]]) if len(vm_edges[i])>0 else (wf_edges)

        self.candidate_taskID = torch.from_numpy(candidate_taskID).to(device)
        self.wf_features = torch.from_numpy(wf_features).to(device)
        self.wf_edges = torch.from_numpy(wf_edges.astype(np.int32)).to(device)
        self.wf_masks = torch.from_numpy(wf_masks).to(device)
        self.wf_batchs = torch.from_numpy(wf_batchs).to(device)

    def wrapper(self, candidate_taskID, wf_features, wf_edges, wf_masks, wf_batchs, vm_features, vm_edges, vm_masks, vm_batchs): 

        if self.normalize:
            wf_features = self.feature_normalize1('features', wf_features)
            vm_features = self.feature_normalize1('vm', vm_features)

       
        for name in ['candidate_taskID', 'wf_features', 'wf_edges', 'vm_batchs']:
            self.__dict__[name] = []        
        totalTasks = 0         
        for i in range(vm_features.shape[0]):
            self.candidate_taskID.append(candidate_taskID + totalTasks)
            temp_wf_fea = deepcopy(wf_features)
            temp_wf_fea[candidate_taskID, 3:] = vm_features[i]
            self.wf_features.append(deepcopy(temp_wf_fea))
            self.wf_edges.append( np.column_stack([wf_edges, vm_edges[i]]) + totalTasks if len(vm_edges[i])>0 else (wf_edges + totalTasks))
            self.vm_batchs.append( np.full((len(wf_features),), i, dtype=np.int64) )
            totalTasks += wf_features.shape[0]

        for name in ['candidate_taskID', 'wf_features', 'vm_batchs']:
            self.__dict__[name] = np.concatenate(self.__dict__[name], axis=0)    
        for name in ['wf_edges']:
            self.__dict__[name] = np.concatenate(self.__dict__[name], axis=1)          
        for name in ['wf_masks', 'wf_batchs']:
            self.__dict__[name] = np.tile(eval(name), vm_features.shape[0])   

        self.candidate_taskID = torch.from_numpy(self.candidate_taskID).to(device)
        self.wf_features = torch.from_numpy(self.wf_features).to(device)
        self.wf_edges = torch.from_numpy(self.wf_edges.astype(np.int32)).to(device)
        self.wf_masks = torch.from_numpy(self.wf_masks).to(device)
        self.wf_batchs = torch.from_numpy(self.wf_batchs).to(device)
        self.vm_batchs = torch.from_numpy(self.vm_batchs).to(device) 

    def batch_process(self, dataList): 
        _state_names = ['candidate_taskID', 'wf_features', 'wf_edges', 'wf_masks', 'wf_batchs']
        for name in _state_names:
            self.__dict__[name] = []

        totalTasks = 0
        # totalNodes = 0
        for i, tensor in enumerate(dataList):
            self.candidate_taskID.append(tensor.candidate_taskID + totalTasks)
            self.wf_features.append(tensor.wf_features)
            self.wf_edges.append(tensor.wf_edges + totalTasks)
            self.wf_masks.append(tensor.wf_masks)
            self.wf_batchs.append(tensor.wf_batchs + i)
            totalTasks += tensor.wf_features.shape[0]
            # self.vm_batchs.append(tensor.vm_batchs + i*tensor.candidate_taskID.shape[0])

        for name in _state_names:
            if name in ['wf_edges']:
                self.__dict__[name] = torch.cat(self.__dict__[name], dim=-1).to(device)
            else: 
                self.__dict__[name] = torch.cat(self.__dict__[name], dim=0).to(device)

        return self

    def clean(self):
        self.candidate_taskID = None
        self.wf_features = None
        self.wf_edges = None
        self.wf_masks = None
        self.wf_batchs = None
        self.vm_features = None
        self.vm_edges = None
        self.vm_masks = None
        self.vm_batchs = None
        
    def feature_normalize(self, data): 
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + + 1e-8)    

    def feature_normalize1(self, name, data):  
        if name == 'wf':
            n = data / np.array(configs.normalize_wf)
        elif name == 'vm':
            n = data / np.array(configs.normalize_vm)
        elif name == 'features':
            n = data / np.array(configs.normalize_features)
        else:
            print('Need to define the maximum value')
        return n 

class Buffer:#(NamedTuple):
    states: list  
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    v_states: list  
    returns: torch.Tensor
    advantages: torch.Tensor

    def __init__(self, state, action, reward, done, log_prob, v_state, return_, advantage_):
        self.states = state
        self.actions = torch.tensor(action, dtype=torch.float32).to(device)
        self.rewards = torch.tensor(reward, dtype=torch.float32).to(device)
        self.dones = torch.tensor(done, dtype=torch.bool).to(device)
        self.log_probs = torch.tensor(log_prob, dtype=torch.float32).to(device)
        self.v_states = v_state
        self.returns = torch.tensor(return_, dtype=torch.float32).to(device)
        self.advantages = torch.tensor(advantage_, dtype=torch.float32).to(device)

class Memory: # state, action, reward, done, prob, val
    def __init__(self, max_len):
        self.state_mb = deque(maxlen=max_len)
        self.action_mb = deque(maxlen=max_len)
        self.reward_mb = deque(maxlen=max_len)
        self.done_mb = deque(maxlen=max_len)   
        self.log_mb = deque(maxlen=max_len)
        # self.state_next_mb = []
        # self.returns = []   
        self.v_state_mb = deque(maxlen=max_len)    
        # self.ep_rewards = 0
        # self.ep_makespan = 0

    def record(self, state, action, reward, done, prob, v_state): 
        self.state_mb.append(state)
        self.action_mb.append(action)
        self.reward_mb.append(reward) 
        self.done_mb.append(done)   
        self.log_mb.append(prob)
        self.v_state_mb.append(v_state)
        # self.state_next_mb.append(staten)
        # self.value_mb.append(val)

    def update(self, states, actions, rewards, dones, probs, v_states): 
        self.state_mb = states
        self.action_mb = actions
        self.reward_mb = rewards 
        self.done_mb = dones   
        self.log_mb = probs
        self.v_state_mb = v_states 
        # self.state_next_mb = statens
        # self.value_mb = vals 

    def clear_memory(self):
        del self.state_mb[:]
        del self.action_mb[:]
        del self.reward_mb[:]
        del self.done_mb[:]  
        del self.log_mb[:]
        del self.v_state_mb[:]

    def compute_returns(self):  # standard
        norm_reward_mb = [i / configs.normalize_rewards for i in self.reward_mb]
        # norm_reward_mb = norm_reward_mb[:len(norm_reward_mb) - configs.window_steps]
        # Initialize a list to store the returns, same length as norm_reward_mb
        discounted_sums = [0] * len(norm_reward_mb)
        # Initialize the last return as 0 (no future rewards after the last step)
        future_return = 0
        for t in reversed(range(len(norm_reward_mb))):
            future_return = norm_reward_mb[t] + configs.gamma * future_return
            discounted_sums[t] = future_return
        # Only keep the returns for len(norm_reward_mb) - configs.window_steps
        return discounted_sums[:len(norm_reward_mb) - configs.window_steps]


    def compute_returns_new(self):  # with slip window
        norm_reward_mb = np.array(self.reward_mb) / configs.normalize_rewards
        # Pre-compute gamma powers
        gamma_powers = np.array([configs.gamma ** i for i in range(configs.window_steps + 1)])
        # Compute discounted sums using convolution
        discounted_sums = np.convolve(norm_reward_mb, gamma_powers[::-1], mode='valid')
        # Slice the valid part based on the original loop
        discounted_sums = discounted_sums[:len(norm_reward_mb) - configs.window_steps]
        # Assign the returns
        return discounted_sums.tolist()

    def compute_gae(self, new_vals):
        # Normalize rewards
        norm_reward_mb = [i / configs.normalize_rewards for i in self.reward_mb]
        # Initialize a list to store the GAE returns, same length as norm_reward_mb
        advantage_mb = [0] * len(norm_reward_mb)
        returns_mb = [0] * len(norm_reward_mb)
        
        # Future return and advantage initialized to 0
        future_advantage = 0
        next_values = 0
        
        # Compute GAE advantage and returns in reverse order
        for t in reversed(range(len(norm_reward_mb))):
            # Delta is the TD residual          
            delta = norm_reward_mb[t] + configs.gamma * next_values - new_vals[t]
            # GAE advantage calculation
            future_advantage = delta + configs.gamma * configs.gae_lambda * future_advantage
            advantage_mb[t] = deepcopy(future_advantage)
            # Return is the sum of advantage and value function
            future_return = new_vals[t] + advantage_mb[t]
            returns_mb[t] = deepcopy(future_return)
            next_values = new_vals[t]   
        
        # Only keep the returns for len(norm_reward_mb) - configs.window_steps
        return returns_mb[:len(norm_reward_mb) - configs.window_steps], advantage_mb[:len(norm_reward_mb) - configs.window_steps]

    def same_length(self):
        self.state_mb = self.state_mb[30:-30]
        self.action_mb = self.action_mb[30:-30]
        self.reward_mb = self.reward_mb[30:-30]
        self.done_mb = self.done_mb[30:-30]      
        self.returns = self.returns[30:-30]  

class RolloutBuffer(Memory):
    def __init__(self, memory, return_, adv_, length):
        super().__init__(0)
        self.returns = deepcopy(return_) 
        self.advantages = deepcopy(adv_) 
        self._state_names = list(self.__dict__.keys())
        # print(self._state_names, '-- length ', length, flush=True)
        for name in self._state_names[:-2]:       
            self.__dict__[name] = list(memory.__dict__[name])[:length] 

    def get(self, batch_size = 64, inorder = False):

        memory_lens = len(self.returns)

        if inorder:
            indices = np.arange(memory_lens)
        else:
            indices = np.random.permutation(memory_lens) 

        start_idx = 0
        while start_idx < memory_lens:
            end_idx = min(start_idx + batch_size, memory_lens)
            yield self.get_samples(indices[start_idx : end_idx])
            start_idx += batch_size

    def get_slice(self, n):
        for name in list(self.__dict__.keys()):
            self.__dict__[name] = self.__dict__[name][:n]

    def get_reversed_slice(self, n):
        for name in list(self.__dict__.keys()):
            self.__dict__[name] = self.__dict__[name][-n:]

    def merge(self, other_buffers):
        for buffer in other_buffers:
            for name in self._state_names:
                self.__dict__[name].extend(buffer.__dict__[name])

    def get_samples(self, batch_inds)-> Buffer:
        data =  [[self.__dict__[name][i] for i in batch_inds] for name in self._state_names]
        return Buffer(*data) 
    
    def update_advantages(self, new_vals):
        self.advantages = deepcopy([a - b for a, b in zip(self.returns, new_vals)])

        # print(self._state_names, '-- length merge ', len(self.returns), flush=True)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_nums, activation_fn=ReLU, dropout=0.0):
        super(MLP, self).__init__()
        layers = []
        # Input layer
        in_dim = input_dim
        for _ in range(layer_nums-1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

class GATlayer(nn.Module):
    def __init__(self, in_chnl, out_chnl, dropout, concat, heads=2):   
        super().__init__()     
        self.dropout = dropout               
        self.conv = GATConv(in_chnl, out_chnl, heads=heads, dropout=dropout, concat=concat) 

    def forward(self, h_node, edge_index):     
        h_node = F.elu(self.conv(F.dropout(h_node, p=self.dropout, training=self.training).float(), edge_index))
        return h_node   ## torch.Size([ ])

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, layers_gat=2, heads=2):
        super().__init__()
        self.layers_gat = layers_gat
        # self.hidden_dim = hidden_dim
        ## GAT conv layers
        self.GAT_layers = torch.nn.ModuleList()
        # init GAT layer
        if layers_gat == 1:
            # only GAT layer
            self.GAT_layers.append(GATlayer(in_dim, hidden_dim, dropout, concat=False, heads=heads))    
        else:
            # first GAT layer
            self.GAT_layers.append(GATlayer(in_dim, hidden_dim, dropout, concat=True, heads=heads))
            # following GAT layers
            for _ in range(layers_gat - 2):
                self.GAT_layers.append(GATlayer(heads * hidden_dim, hidden_dim, dropout, concat=True, heads=heads))
            # last GAT layer
            self.GAT_layers.append(GATlayer(heads * hidden_dim, hidden_dim, dropout, concat=False, heads=1))

    def forward(self, x, edge_index):

        h_node = self.GAT_layers[0](x, edge_index) 
        for layer in range(1, self.layers_gat):
            h_node = self.GAT_layers[layer](h_node, edge_index)

        return h_node   


class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers_gin=2):
        super(GIN, self).__init__()
        self.layer_gin = layers_gin
        self.GIN_layers = torch.nn.ModuleList()
        self.GIN_layers.append(
            GINConv(
                Sequential(Linear(in_dim, hidden_dim),
                           nn.BatchNorm1d(hidden_dim),
                           ReLU(),
                           Linear(hidden_dim, hidden_dim)),
                eps=0,
                train_eps=False,
                aggr='mean',
                flow="source_to_target")
        )
        # rest gin layers
        for _ in range(layers_gin - 1):
            self.GIN_layers.append(
                GINConv(
                    Sequential(Linear(hidden_dim, hidden_dim),
                               nn.BatchNorm1d(hidden_dim),
                               ReLU(),
                               Linear(hidden_dim, hidden_dim)),
                    eps=0,
                    train_eps=False,
                    aggr='mean',
                    flow="source_to_target")
            )       

    def forward(self, x, edge_index):
        edge_index = edge_index.long()
        h_node = self.GIN_layers[0](x.float(), edge_index) 
        for layer in range(1, self.layer_gin):
            h_node = F.relu(h_node)
            h_node = self.GIN_layers[layer](h_node, edge_index)
        return h_node


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout, layers_attn=2, heads=2):  
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead= 2, dim_feedforward=ff_dim, dropout = dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=layers_attn)        

    def forward(self, x, padding_mask=None):
        if padding_mask is not None:
            output = self.encoder(x, src_key_padding_mask=padding_mask)
        else:
            output = self.encoder(x)
        return output
    

class Critic(nn.Module):
    def __init__(self,
                 input_dim_wf,
                 input_dim_vm,
                 hidden_dim,                 
                 gnn_layers,
                 atten_layers,                  
                 mlp_layers,             
                 heads=1,
                 dropout=0.0,
                 ):
        super().__init__()    
        self.gnn_layers = gnn_layers
        self.mlp_layers = mlp_layers
        self.atten_layers = atten_layers
        # self.vmNum = configs.each_vm_type_num * configs.vm_types

        ## GNN
        self.embedding_wf = GAT(in_dim=input_dim_wf+input_dim_vm, hidden_dim=hidden_dim, dropout=dropout, layers_gat=gnn_layers, heads=heads)                        

        # Self-attention
        if atten_layers> 0:
            self.attention = SelfAttention(embed_dim = hidden_dim, ff_dim = hidden_dim*4, dropout = dropout , layers_attn=atten_layers, heads=heads)

        # Critic
        self.critic = MLP(hidden_dim, hidden_dim, 1, mlp_layers) 
        # sum(p.numel() for p in model.parameters())

    def prepare_input(self, A, mask_A, batch_index, candidate):
        # 根据 batch_index 将 A 和 mask_A 重新组织成 (batch_size, seq_length, embed_dim) 的形状
        batch_size = batch_index.max().item() + 1
        seq_lengths = torch.bincount(batch_index)
        max_seq_length = seq_lengths.max().item()
        embed_dim = A.size(1)
        output_A = torch.zeros(batch_size, max_seq_length, embed_dim)
        output_mask_A = torch.zeros(batch_size, max_seq_length, dtype=torch.bool)
        original_idx = torch.zeros((batch_size*max_seq_length))
        cum_sum = 0
        for i in range(batch_size):
            idx = (batch_index == i).nonzero().squeeze()
            output_A[i, :len(idx)] = A[idx]
            output_mask_A[i, :len(idx)] = mask_A[idx]
            original_idx[cum_sum:cum_sum+len(idx)] = idx
            cum_sum += max_seq_length 
        # 从 original_idx 中搜索 candidate
        candidate_positions = []
        for c in candidate:
            pos = torch.where(original_idx == c)[0][0]
            candidate_positions.append(pos.item())
        
        return output_A, output_mask_A, torch.tensor(candidate_positions)

    # @monitor
    # @profile
    def forward(self,
                state_wf,       
                edge_index_wf,            
                mask_wf,
                batch_wf,
                candidate_task_index, 
                deterministic = False,          
                ):
        
        if configs.require_undirected == 1:
            edge_index_wf = torch.cat((edge_index_wf, edge_index_wf.flip(0)), dim=-1)
            # edge_index_wf = torch.unique(edge_index_wf, dim=-1)

        wf_task_embed = self.embedding_wf(state_wf, edge_index_wf)  # (360, 32)

        if self.atten_layers > 0:
            input_data, input_mask, candidate_idx  =  self.prepare_input(wf_task_embed, mask_wf, batch_wf, candidate_task_index)
            output_atten_task = self.attention(input_data, padding_mask=~input_mask)
            masked_sum = torch.sum(output_atten_task * input_mask.unsqueeze(-1), dim=1)
            mask_sum = torch.sum(input_mask, dim=1, keepdim=True)
            average_task_embed = masked_sum / mask_sum  # (batch_size, hidden_dim)
            # output_atten_task = output_atten_task.reshape(-1, wf_task_embed.shape[-1])
            # candidate_tasks = output_atten_task[candidate_idx]

        else:
            average_task_embed = global_mean_pool(wf_task_embed[mask_wf], batch_wf[mask_wf]) 
            # candidate_tasks = wf_task_embed[candidate_task_index]

        # Calculate prob
        values = self.critic(average_task_embed)    # (144, 1)

        return values.squeeze()       


class Actor(nn.Module):
    def __init__(self,
                 input_dim_wf,
                 input_dim_vm,
                 hidden_dim,                 
                 gnn_layers,
                 mlp_layers,           
                 heads=1,
                 dropout=0.0,
                 activate_fn = 'relu',
                 embedding_type = 'gat',
                 ):
        super().__init__()
        self.gnn_layers = gnn_layers
        self.mlp_layers = mlp_layers
        # self.atten_layers = atten_layers
        self.vmNum = configs.each_vm_type_num * configs.vm_types
        
        ## GNN
        if embedding_type == 'gat':
            self.embedding_wf = GAT(in_dim=input_dim_wf+input_dim_vm, hidden_dim=hidden_dim, dropout=dropout, layers_gat=gnn_layers, heads=heads)
        # self.embedding_vm = GAT(in_dim=input_dim_vm, hidden_dim=hidden_dim, dropout=dropout, layers_gat=gnn_layers, heads=heads)                            
        elif embedding_type == 'gin':
            self.embedding_wf = GIN(in_dim=input_dim_wf+input_dim_vm, hidden_dim=hidden_dim, layers_gin=gnn_layers)

        # Policy
        if configs.require_mean == 1:
            in_dim = hidden_dim*2 
        else:
            in_dim = hidden_dim
        self.actor = MLP(in_dim, hidden_dim, 1, mlp_layers) 

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

    def forward(self,
                state_wf,      #
                state_vm,      # (18, 5)           
                edge_index_wf,        #
                edge_index_vm,        # (18, 2)           
                mask_wf,
                mask_vm,
                batch_wf,
                batch_vm,
                candidate_task_index,  ## (batch_size, n_j)  
                deterministic = False,      
                # require_meanPool = True,        
                ):

        edge_index_wf = torch.cat((edge_index_wf, edge_index_wf.flip(0)), dim=-1)
        # if configs.require_undirected == 1:
        #     edge_index_wf = torch.cat((edge_index_wf, edge_index_wf.flip(0)), dim=-1)
            # edge_index_wf = torch.unique(edge_index_wf, dim=-1)

        wf_task_embed = self.embedding_wf(state_wf, edge_index_wf)  # (360, 32)
        hidden_dim = wf_task_embed.shape[-1] 
        candidate_tasks = wf_task_embed[candidate_task_index]  # (18, 32)
        candidate_tasks = candidate_tasks.reshape(-1, self.vmNum, hidden_dim)

        graph_embed = candidate_tasks
        # if configs.require_mean == 1:
        #     average_vm_embed = global_mean_pool(wf_task_embed[mask_wf], batch_vm[mask_wf]).reshape(-1, self.vmNum, hidden_dim)  # (1, 18, 32)
        #     graph_embed = torch.cat((candidate_tasks, average_vm_embed), dim=-1)    # (1, 18, 64)
        # else:
        #     graph_embed = candidate_tasks # (1, 18, 32)

        # Calculate prob
        candidate_scores = self.actor(graph_embed)

        pi = F.softmax(candidate_scores, dim=1)     # (batch_Size, 18 ,1)
        dist = Categorical(probs=pi.squeeze())
        if deterministic is True:
            arr = pi.squeeze(-1)
            max_values, _ = torch.max(arr, dim=-1)
            actions_id = torch.zeros(arr.size(0), dtype=torch.int64)
            for i in range(arr.size(0)):
                max_positions = torch.nonzero(arr[i] > max_values[i] -1e-5, as_tuple=False).squeeze(1)
                random_choice = torch.randint(len(max_positions), (1,)).item()
                actions_id[i] = max_positions[random_choice]            
        else:
            actions_id = dist.sample()
        log_prob = dist.log_prob(actions_id)

        return actions_id, log_prob, dist.entropy().mean()

    def eval_actions(self,
                state_wf,      ## (batch_szie^2, 2)
                state_vm,                 
                edge_index_wf,        ## (batch_size^2, batch_size^2)
                edge_index_vm,        ## (batch_size^2, batch_size^2)                
                mask_wf,
                mask_vm,
                batch_wf,
                batch_vm,
                candidate_task_index,  ## (batch_size, n_j)       
                actions,      
                ):
        
        edge_index_wf = torch.cat((edge_index_wf, edge_index_wf.flip(0)), dim=-1)

        wf_task_embed = self.embedding_wf(state_wf, edge_index_wf) 
        hidden_dim = wf_task_embed.shape[-1]         
        candidate_tasks = wf_task_embed[candidate_task_index]  # (18, 32)
        candidate_tasks = candidate_tasks.reshape(-1, self.vmNum, hidden_dim)

        graph_embed = candidate_tasks 

        # Calculate prob
        candidate_scores = self.actor(graph_embed)

        pi = F.softmax(candidate_scores, dim=1) 
        dist = Categorical(probs=pi.squeeze())
        log_prob = dist.log_prob(actions.clone().detach())
        entropy = dist.entropy().mean()

        return log_prob, entropy

    def eval_dists(self,
                state_wf,      ## (batch_szie^2, 2)
                state_vm,                 
                edge_index_wf,        ## (batch_size^2, batch_size^2)
                edge_index_vm,        ## (batch_size^2, batch_size^2)                
                mask_wf,
                mask_vm,
                batch_wf,
                batch_vm,
                candidate_task_index,  ## (batch_size, n_j)            
                ):
        
        edge_index_wf = torch.cat((edge_index_wf, edge_index_wf.flip(0)), dim=-1)

        wf_task_embed = self.embedding_wf(state_wf, edge_index_wf) 
        hidden_dim = wf_task_embed.shape[-1]         
        candidate_tasks = wf_task_embed[candidate_task_index]  # (18, 32)
        candidate_tasks = candidate_tasks.reshape(-1, self.vmNum, hidden_dim)

        graph_embed = candidate_tasks
        candidate_scores = self.actor(graph_embed)

        pi = F.softmax(candidate_scores, dim=1) 
        dist = Categorical(probs=pi.squeeze())
        entropy = dist.entropy().mean()

        return pi.squeeze(), entropy

class REINFORCE:
    def __init__(self,
                 input_dim_wf,
                 input_dim_vm,
                 hidden_dim,
                 gnn_layers,
                 mlp_layers,
                 heads,
                 dropout,
                 activate_fn,                
                 ):

        self.actor = Actor(input_dim_wf,
                            input_dim_vm,           
                            hidden_dim,
                            gnn_layers,
                            mlp_layers,                                                            
                            heads,
                            dropout,     
                            activate_fn,   
                            ).to(device)     

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=configs.lr_a)

    def train_HEFT(self, bufferdata):
        # state_mb, action_mb, reward_mb
        cross_losses = []
        batch_states = BatchGraph(configs.normalize)
        criterion = nn.CrossEntropyLoss()
        memory_lens = len(bufferdata[0])
        for _ in range(configs.n_epochs):#int(configs.max_updates/2)):
            indices = np.random.permutation(np.arange(configs.warmup_steps, memory_lens-configs.warmup_steps)) 
            start_idx = 0
            cross_loss = []
            while start_idx < len(indices): # - configs.warmup_steps*2):
                end_idx = min(start_idx + configs.batch_size, len(indices))

                temp_states = []
                for k in indices[start_idx:end_idx]:
                    state_list = bufferdata[0][k]
                    batch_states.wrapper(*state_list)   
                    temp_states.append(deepcopy(batch_states))
                temp_actions = torch.tensor([bufferdata[1][k] for k in indices[start_idx:end_idx]], dtype=torch.int32).to(device)
                if temp_states == []:
                    a = 1
                batch_states = BatchGraph(configs.normalize).batch_process(temp_states)

                dists = self.actor.eval_dists(state_wf = batch_states.wf_features,     
                                    state_vm = batch_states.vm_features,                 
                                    edge_index_wf = batch_states.wf_edges,    
                                    edge_index_vm = batch_states.vm_edges,   
                                    mask_wf = batch_states.wf_masks,
                                    mask_vm = batch_states.vm_masks,
                                    batch_wf = batch_states.wf_batchs,
                                    batch_vm = batch_states.vm_batchs,
                                    candidate_task_index = batch_states.candidate_taskID
                                    ) 
                    
                loss = criterion(dists, temp_actions.long())
                cross_loss.append(loss.item())
                self.optimizer_actor.zero_grad()
                loss.backward()
                self.optimizer_actor.step()
                if configs.require_clip_value > 1: 
                    for param in self.actor.parameters():
                        param.data = torch.clamp(param.data, -configs.require_clip_value, configs.require_clip_value)            
                start_idx += configs.batch_size
            cross_losses.append(np.mean(cross_loss))
        return cross_losses       

    def train(self, bufferdata):
        returns = torch.tensor([m.returns/configs.normalize_rewards for m in bufferdata], dtype=torch.float32)
        returns = (returns - returns.mean()) / (torch.std(returns, unbiased=False) + np.finfo(np.float32).eps.item())
        memory_lens =len(bufferdata[0].state_mb)
        M_losses = []
        for _ in range(configs.epochs_a):

            M_loss = []
            indices = np.random.permutation(np.arange(memory_lens)) 
            start_idx = 0
            while start_idx < memory_lens:
                end_idx = min(start_idx + configs.batch_size, memory_lens)
                losses = []
                for i,m in enumerate(bufferdata):      

                    temp_states = [m.state_mb[k] for k in indices[start_idx:end_idx]]
                    temp_actions = torch.tensor([m.action_mb[k] for k in indices[start_idx:end_idx]], dtype=torch.int32).to(device)
                    batch_states = BatchGraph(configs.normalize).batch_process(temp_states)
                    log_probs,_ = self.actor.eval_actions(state_wf = batch_states.wf_features,     
                                        state_vm = batch_states.vm_features,                 
                                        edge_index_wf = batch_states.wf_edges,    
                                        edge_index_vm = batch_states.vm_edges,   
                                        mask_wf = batch_states.wf_masks,
                                        mask_vm = batch_states.vm_masks,
                                        batch_wf = batch_states.wf_batchs,
                                        batch_vm = batch_states.vm_batchs,
                                        candidate_task_index = batch_states.candidate_taskID,
                                        actions = temp_actions) 
                        
                    gt = returns[i]        
                    loss = - (log_probs * gt).sum()   
                    losses.append(loss)                 

                self.optimizer_actor.zero_grad()
                mean_loss = torch.stack(losses).mean()
                M_loss.append(mean_loss.item())
                mean_loss.backward()
                self.optimizer_actor.step()
                if configs.require_clip_value > 1: 
                    for param in self.actor.parameters():
                        param.data = torch.clamp(param.data, -configs.require_clip_value, configs.require_clip_value)            
                start_idx += configs.batch_size

            M_losses.append(np.mean(M_loss))

        return M_losses        

class PPO:
    def __init__(self,
                 input_dim_wf,
                 input_dim_vm,
                 hidden_dim,
                 c_hidden_dim,
                 gnn_layers,
                 atten_layers,
                 mlp_layers,
                 heads,
                 dropout,
                 activate_fn,                
                 ):

        self.actor = Actor(input_dim_wf,
                            input_dim_vm,           
                            hidden_dim,
                            gnn_layers,
                            mlp_layers,                                                            
                            heads,
                            dropout,     
                            activate_fn,   
                            ).to(device)
        
        self.critic = Critic(input_dim_wf,
                            input_dim_vm,
                            c_hidden_dim,                 
                            gnn_layers,
                            atten_layers,                             
                            mlp_layers,               
                            heads,
                            dropout,
                            ).to(device)          

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=configs.lr_a)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=configs.lr_c)

    def train_HEFT(self, bufferdata):
        # state_mb, action_mb, reward_mb
        cross_losses = []
        batch_states = BatchGraph(configs.normalize)
        criterion = nn.CrossEntropyLoss()
        memory_lens = len(bufferdata[0])
        for _ in range(configs.n_epochs): 
            indices = np.random.permutation(np.arange(configs.warmup_steps, memory_lens-configs.warmup_steps)) 
            start_idx = 0
            cross_loss = []
            while start_idx < len(indices):  
                end_idx = min(start_idx + configs.batch_size, len(indices))

                temp_states = []
                for k in indices[start_idx:end_idx]:
                    state_list = bufferdata[0][k]
                    batch_states.wrapper(*state_list)   
                    temp_states.append(deepcopy(batch_states))
                temp_actions = torch.tensor([bufferdata[1][k] for k in indices[start_idx:end_idx]], dtype=torch.int32).to(device)

                batch_states = BatchGraph(configs.normalize).batch_process(temp_states)

                dists = self.actor.eval_dists(state_wf = batch_states.wf_features,     
                                    state_vm = batch_states.vm_features,                 
                                    edge_index_wf = batch_states.wf_edges,    
                                    edge_index_vm = batch_states.vm_edges,   
                                    mask_wf = batch_states.wf_masks,
                                    mask_vm = batch_states.vm_masks,
                                    batch_wf = batch_states.wf_batchs,
                                    batch_vm = batch_states.vm_batchs,
                                    candidate_task_index = batch_states.candidate_taskID
                                    ) 
                    
                loss = criterion(dists, temp_actions.long())
                cross_loss.append(loss.item())
                self.optimizer_actor.zero_grad()
                loss.backward()
                self.optimizer_actor.step()
                if configs.require_clip_value > 1: 
                    for param in self.actor.parameters():
                        param.data = torch.clamp(param.data, -configs.require_clip_value, configs.require_clip_value)            
                start_idx += configs.batch_size
            cross_losses.append(np.mean(cross_loss))
        return cross_losses    

    def train_critic(self, bufferdata):

        rmse_losses, mre_losses = [], []

        for _ in range(configs.epochs_c):
            value_loss,rate_loss = [], []
            for rollout_data in bufferdata.get(configs.batch_size): 

                batch_states = BatchGraph(configs.normalize).batch_process(rollout_data.v_states) 
                vals = self.critic(state_wf = batch_states.wf_features,              
                                edge_index_wf = batch_states.wf_edges,       
                                mask_wf = batch_states.wf_masks,
                                batch_wf = batch_states.wf_batchs,
                                candidate_task_index = batch_states.candidate_taskID,
                                deterministic = False)            

                v_loss = F.mse_loss(rollout_data.returns, vals)
                value_loss.append(v_loss.item() * vals.shape[0]) 
                rate_loss.append( torch.sum(torch.abs((rollout_data.returns - vals) / rollout_data.returns) ).item() )

                self.optimizer_critic.zero_grad()
                v_loss.backward() 
                self.optimizer_critic.step()

            rmse_losses.append(np.sqrt(np.sum(value_loss)/ len(bufferdata.state_mb)))
            mre_losses.append(100*np.sum(rate_loss)/ len(bufferdata.state_mb)) 

        batch_states = BatchGraph(configs.normalize).batch_process(bufferdata.v_state_mb)
        with torch.no_grad():
            values = self.critic(state_wf = batch_states.wf_features,              
                            edge_index_wf = batch_states.wf_edges,       
                            mask_wf = batch_states.wf_masks,
                            batch_wf = batch_states.wf_batchs,
                            candidate_task_index = batch_states.candidate_taskID,
                            deterministic = False)


        return values.tolist(), (rmse_losses, mre_losses)

    def train_actor(self, bufferdata):

        # pre_l2_norm = self.pre_grad
        pg_losses, entropy_losses, grad_changes = [], [], []  
        for _ in range(configs.epochs_a):
            pg_loss, entropy_loss = [], []
            for rollout_data in bufferdata.get(configs.batch_size):           

                batch_states = BatchGraph(configs.normalize).batch_process(rollout_data.states) 
                logprobs, ent_loss = self.actor.eval_actions(state_wf = batch_states.wf_features,     
                                state_vm = batch_states.vm_features,                 
                                edge_index_wf = batch_states.wf_edges,    
                                edge_index_vm = batch_states.vm_edges,   
                                mask_wf = batch_states.wf_masks,
                                mask_vm = batch_states.vm_masks,
                                batch_wf = batch_states.wf_batchs,
                                batch_vm = batch_states.vm_batchs,
                                candidate_task_index = batch_states.candidate_taskID,
                                actions = rollout_data.actions) 

                advantages = rollout_data.advantages    ## A = r(i.e.,Q) - v

                ratio = torch.exp(logprobs - rollout_data.log_probs)   
                surr1 = advantages * ratio     
                surr2 = advantages * torch.clamp(ratio, 1 - configs.eps_clip, 1 + configs.eps_clip)
                p_loss = torch.min(surr1, surr2).mean()
                         
                ent_loss = torch.mean(ent_loss) 
                entropy_loss.append(ent_loss.item())

                if configs.entropy_control == 0:
                    e_coef = 0
                elif entropy_loss[-1] < configs.entropy_min:
                    e_coef = configs.entloss_coef
                    self.entropy_count +=1
                elif entropy_loss[-1] > configs.entropy_max:
                    e_coef = -configs.entloss_coef
                    self.entropy_count +=1
                else:
                    e_coef = 0
 
                loss = -p_loss - e_coef * ent_loss
                pg_loss.append(loss.item())  

                self.optimizer_actor.zero_grad()
                loss.backward() 
                l2_norm = []
                for _, param in self.actor.named_parameters():
                    if param.grad is not None:
                        # L2 范数
                        grad_l2_norm = torch.norm(param.grad)
                        l2_norm.append(grad_l2_norm.item())
                l2_norm_mean = np.mean(l2_norm)

                if configs.grad_control ==1:
                    if l2_norm_mean<=0.075 and (self.pre_grad_max==0 or l2_norm_mean<=self.pre_grad_max):
                        grad_changes.append(deepcopy(l2_norm_mean))
                        self.optimizer_actor.step()  
                    else:
                       self.grad_count +=1
                else:
                    grad_changes.append(deepcopy(l2_norm_mean))
                    self.optimizer_actor.step()                      

                for param in self.actor.parameters():
                    param.data = torch.clamp(param.data, -configs.require_clip_value, configs.require_clip_value) 

            pg_losses.append(np.mean(pg_loss))
            entropy_losses.append(np.mean(entropy_loss))
            # clip_fractions.append(np.mean(clip_frac))

        if len(grad_changes) == 0:
            self.pre_grad_max = 0
            grad_changes = [0]
        else:
            self.pre_grad_max = np.mean(grad_changes)+ np.std(grad_changes) # np.max([ np.mean(grad_changes)*2 , np.mean(grad_changes) + np.std(grad_changes)])

        return (pg_losses, entropy_losses, grad_changes)


    def train(self, bufferdata):    # offline train actor+critic
        
        start_idx = 0
        values = []
        while start_idx <  len(bufferdata.v_state_mb):
            end_idx = min(start_idx + configs.batch_size, len(bufferdata.v_state_mb))
            batch_states = BatchGraph(configs.normalize).batch_process(bufferdata.v_state_mb[start_idx:end_idx]) 
            vals = self.critic(state_wf = batch_states.wf_features,              
                            edge_index_wf = batch_states.wf_edges,       
                            mask_wf = batch_states.wf_masks,
                            batch_wf = batch_states.wf_batchs,
                            candidate_task_index = batch_states.candidate_taskID,
                            deterministic = False) 
            values.extend(vals.tolist())
            start_idx += configs.batch_size                    
        bufferdata.update_advantages(values)
        actor_samples = deepcopy(bufferdata)


        pg_losses, entropy_losses, grad_changes = [], [], []
        va_losses, all_loss, mre_losses = [], [], []  
        for _ in range(configs.n_epochs):
            pg_loss, entropy_loss, va_loss, rate_loss = [], [], [], []  
            for rollout_data in actor_samples.get(configs.batch_size):      

                batch_states = BatchGraph(configs.normalize).batch_process(rollout_data.states) 
                logprobs, ent_loss = self.actor.eval_actions(state_wf = batch_states.wf_features,    
                                state_vm = batch_states.vm_features,                 
                                edge_index_wf = batch_states.wf_edges,    
                                edge_index_vm = batch_states.vm_edges,   
                                mask_wf = batch_states.wf_masks,
                                mask_vm = batch_states.vm_masks,
                                batch_wf = batch_states.wf_batchs,
                                batch_vm = batch_states.vm_batchs,
                                candidate_task_index = batch_states.candidate_taskID,
                                actions = rollout_data.actions) 

                if configs.normalize_advantage:
                    advantages = (rollout_data.advantages - rollout_data.advantages.mean()) / (rollout_data.advantages.std() + 1e-8)   
                else:
                    advantages = rollout_data.advantages
                                ## A = r(i.e.,Q) - v
                ratio = torch.exp(logprobs - rollout_data.log_probs)   
                surr1 = advantages * ratio     
                surr2 = advantages * torch.clamp(ratio, 1 - configs.eps_clip, 1 + configs.eps_clip)
                p_loss = torch.min(surr1, surr2).mean()
                pg_loss.append(p_loss.item())           

                batch_states = BatchGraph(configs.normalize).batch_process(rollout_data.v_states) 
                vals = self.critic(state_wf = batch_states.wf_features,              
                                edge_index_wf = batch_states.wf_edges,       
                                mask_wf = batch_states.wf_masks,
                                batch_wf = batch_states.wf_batchs,
                                candidate_task_index = batch_states.candidate_taskID,
                                deterministic = False)       
                v_loss = F.mse_loss(rollout_data.returns, vals)    #  PPO：F.mse_loss(rollout_data.returns, vals.squeeze())                     
                va_loss.append(v_loss.item()) 
                rate_loss.append( torch.sum(torch.abs((rollout_data.returns - vals) / rollout_data.returns) ).item() )

                ent_loss = torch.mean(ent_loss) 
                entropy_loss.append(ent_loss.item())

                if configs.entropy_control == 0:
                    e_coef = 0
                elif entropy_loss[-1] < configs.entropy_min:
                    e_coef = configs.entloss_coef
                    self.entropy_count +=1
                elif entropy_loss[-1] > configs.entropy_max:
                    e_coef = -configs.entloss_coef
                    self.entropy_count +=1
                else:
                    e_coef = 0

                if self.update_idx > configs.warmup_critic:
                    p_coef = 1
                else:
                    p_coef = 0

                loss = -p_coef*p_loss + configs.vloss_coef * v_loss -  e_coef* ent_loss

                all_loss.append(loss.item())
                self.optimizer.zero_grad()
                loss.mean().backward()   

                l2_norm = []
                for _, param in self.actor.named_parameters():
                    if param.grad is not None:
                        # L2 范数
                        grad_l2_norm = torch.norm(param.grad)
                        l2_norm.append(grad_l2_norm.item())
                l2_norm_mean = np.mean(l2_norm)
                if configs.grad_control ==1:
                    if l2_norm_mean<=0.075 and (self.pre_grad_max==0 or l2_norm_mean<=self.pre_grad_max):
                        grad_changes.append(deepcopy(l2_norm_mean))
                        self.optimizer_actor.step()  
                    else:
                        self.grad_count +=1
                    #     grad_changes.append(0)
                else:
                    grad_changes.append(deepcopy(l2_norm_mean))
                    self.optimizer.step()  

                # if configs.require_clip_value > 1: # 10
                for param in self.actor.parameters():
                    param.data = torch.clamp(param.data, -configs.require_clip_value, configs.require_clip_value) 

            pg_losses.append(np.mean(pg_loss))
            entropy_losses.append(np.mean(entropy_loss))
            va_losses.append(np.mean(va_loss))
            mre_losses.append(100*np.sum(rate_loss)/ len(bufferdata.state_mb))

            # clip_fractions.append(np.mean(clip_frac))

        if len(grad_changes) == 0:
            self.pre_grad_max = 0
            grad_changes = [0]
        else:
            self.pre_grad_max = np.mean(grad_changes)+ np.std(grad_changes)

        return (all_loss, pg_losses, entropy_losses), (va_losses, mre_losses, grad_changes)

