from copy import deepcopy
import torch, pickle
import time, random
import numpy as np
from config.Params import configs
import torch.nn as nn
from policy.actor3 import PPO, BatchGraph
from env.workflow_scheduling_v3.simulator_wf import WFEnv
from env.workflow_scheduling_v3.lib.poissonSampling import sample_poisson_shape
# from joblib import Parallel, delayed    # parallel version

device = torch.device(configs.device)
# file_writing_a = './logs/actor_log_' + str(configs.epochs_c) + '_' + str(configs.lr_c) + '_' + str(configs.window_steps) + '.npy'
# file_writing_c = './logs/critic_log_' + str(configs.epochs_c) + '_' + str(configs.lr_c) + '_' + str(configs.window_steps) + '.npy'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)    

def validation_H(i, args):
    set_seed(args.env_seed)
    args.indEVALindex = i
    env = WFEnv(args.env_name, args, False)
    state_list = env.resetGP()
    while True:
        action = env.HEFT()
        state_list, _, done = env.stepGP(action)   ## 比env.rest()多了reward, done
        if done:  
            return np.mean(env.all_flowTime)   

def validation_Org(i, args, model, deterministic):
    args.indEVALindex = i
    envs = WFEnv(args.env_name, args, False)
    state_list = envs.reset()
    batch_states = BatchGraph(args.normalize)
    while True:
        with torch.no_grad():
            batch_states.wrapper(*state_list)   # 处理(s,a)
            action, _,_ = model(state_wf = batch_states.wf_features,      ## (batch_szie^2, 2)
                                state_vm = batch_states.vm_features,                 
                                edge_index_wf = batch_states.wf_edges,    
                                edge_index_vm = batch_states.vm_edges,   
                                mask_wf = batch_states.wf_masks,
                                mask_vm = batch_states.vm_masks,
                                batch_wf = batch_states.wf_batchs,
                                batch_vm = batch_states.vm_batchs,
                                candidate_task_index = batch_states.candidate_taskID,
                                deterministic = deterministic)
        state_list, _, done = envs.step(action.item())  
        if done:  
            return np.mean(envs.all_flowTime)   
        
def append_to_nested_lists(nested_lists, new_lists):
    for i in range(len(nested_lists)):
        nested_lists[i].extend(new_lists[i])

def main():

    # Load dataset
    record = 1e10
    wf_types = 4  
    set_seed(configs.env_seed)
    configs.valid_dataset = np.load('./validation_data/validation_instance_2024.npy').reshape((1,-1, configs.wf_num)) [:, :(configs.valid_num+ configs.num_envs)]
    configs.GENindex = 0
    configs.indEVALindex = 0
    configs.arr_times = sample_poisson_shape(configs.arr_rate, configs.valid_dataset.shape)

    configs.train_dataset = np.random.randint(0,wf_types,(configs.max_updates+1, configs.num_envs, configs.wf_num ))

    with open('./validation_data/HEFT/HEFT_memory_500_{}_{}_{}.pkl'.format(configs.vm_types, configs.each_vm_type_num ,configs.arr_rate ), 'rb') as file:
        bufferdata = pickle.load(file)[:3]      # state_mb, action_mb, reward_mb, done_mb, graph_nodes    

    # non-parallel
    meanFlowTimes = []
    for t in range(configs.valid_num):
        meanFlowTime = validation_H(t, configs)
        meanFlowTimes.append(meanFlowTime)

    # # parallel
    # meanFlowTimes = Parallel(n_jobs=-1)(delayed(validation_H)( t, configs ) for t in range(configs.valid_num)) 

    t1 = time.time()
    print('Vlidation at HEFT: mean_flowtime_deterministic: {:.3f}+/-{:.3f}\t time_elapsed: {:.3f}'.\
            format(np.mean(meanFlowTimes), np.std(meanFlowTimes), (t1 - total1)/3600), flush=True)    

    # Build policy
    set_seed(configs.algo_seed)
    algos = PPO(input_dim_wf = configs.input_dim_wf,
                    input_dim_vm= configs.input_dim_vm,           
                    hidden_dim= configs.hidden_dim,
                    c_hidden_dim= configs.c_hidden_dim,
                    gnn_layers= configs.gnn_layers,
                    atten_layers = configs.atten_layers,  
                    mlp_layers= configs.mlp_layers,                                                                               
                    heads= configs.heads,
                    dropout= configs.dropout,     
                    activate_fn = configs.activate_fn,   
                    )

    # non-parallel
    meanFlowTimes = []
    for t in range(configs.valid_num):
        meanFlowTime = validation_Org(t, configs, algos.actor, True)
        meanFlowTimes.append(meanFlowTime)

    # # parallel
    # meanFlowTimes = Parallel(n_jobs=-1)(delayed(validation_Org)( t, configs, algos.actor, True ) for t in range(configs.valid_num)) 

    if np.mean(meanFlowTimes) < record:
        # torch.save(algos.actor.state_dict(), './logs/actor_{}.pth'.format(str(i_update + 1)) ) 
        record = deepcopy(np.mean(meanFlowTimes))
    t1 = time.time()
    print('Vlidation at update-{}: mean_flowtime_deterministic: {:.3f}+/-{:.3f}\t record: {:.3f}\t time_elapsed: {:.3f}'.\
            format(str(0), np.mean(meanFlowTimes), np.std(meanFlowTimes), record, (t1 - total1)/3600), flush=True)

    cross_losses = []
    ent_losses = []
    batch_states = BatchGraph(configs.normalize)
    criterion = nn.CrossEntropyLoss()
    memory_lens = len(bufferdata[0])
    for i_update in range(configs.max_updates):   
        indices = np.random.permutation(np.arange(configs.warmup_steps, memory_lens-configs.warmup_steps)) 
        start_idx = 0
        cross_loss = []
        ent_loss = []
        while start_idx < len(indices): 
            end_idx = min(start_idx + configs.batch_size, len(indices))

            temp_states = []
            for k in indices[start_idx:end_idx]:
                state_list = bufferdata[0][k]
                batch_states.wrapper(*state_list)   
                temp_states.append(deepcopy(batch_states))
            temp_actions = torch.tensor([bufferdata[1][k] for k in indices[start_idx:end_idx]], dtype=torch.int32).to(device)

            batch_states = BatchGraph(configs.normalize).batch_process(temp_states)

            dists, e_loss = algos.actor.eval_dists(state_wf = batch_states.wf_features,     
                                state_vm = batch_states.vm_features,                 
                                edge_index_wf = batch_states.wf_edges,    
                                edge_index_vm = batch_states.vm_edges,   
                                mask_wf = batch_states.wf_masks,
                                mask_vm = batch_states.vm_masks,
                                batch_wf = batch_states.wf_batchs,
                                batch_vm = batch_states.vm_batchs,
                                candidate_task_index = batch_states.candidate_taskID
                                ) 

            e_loss = torch.mean(e_loss) # - ent_loss.clone()
            ent_loss.append(e_loss.item())

            p_loss = criterion(dists, temp_actions.long())

            if configs.entloss_coef>0:
                loss = p_loss - configs.entloss_coef*e_loss
            else:
                loss = p_loss
            cross_loss.append(p_loss.item())
            algos.optimizer_actor.zero_grad()
            loss.backward()
            algos.optimizer_actor.step()
            if configs.require_clip_value > 1: 
                for param in algos.actor.parameters():
                    param.data = torch.clamp(param.data, -configs.require_clip_value, configs.require_clip_value)            
            start_idx += configs.batch_size
        cross_losses.append(cross_loss)
        ent_losses.append(ent_loss)
        t1=time.time()
        print('Episode-{}: p_loss: {:.3f}\t e_loss: {:.3f}\t time_elapsed: {:.2f}'.format(str(i_update+1),\
                            np.mean(cross_loss), np.mean(ent_loss),(t1 - total1)/3600), flush=True)        
        # np.save('./logs/cross_{}-{}-{}.npy'.format(configs.vm_types, configs.each_vm_type_num ,configs.arr_rate), cross_losses)
        # np.save('./logs/entropy_{}-{}-{}.npy'.format(configs.vm_types, configs.each_vm_type_num ,configs.arr_rate), ent_losses)

        if (i_update + 1) % configs.log_interval == 0:

            # non-parallel
            meanFlowTimes = []
            for t in range(configs.valid_num):
                meanFlowTime = validation_Org(t, configs,algos.actor, True)
                meanFlowTimes.append(meanFlowTime)

            # # parallel
            # meanFlowTimes = Parallel(n_jobs=-1)(delayed(validation_Org)( t, configs,algos.actor, True ) for t in range(configs.valid_num))
            
            if np.mean(meanFlowTimes) < record:
                torch.save(algos.actor.state_dict(), './logs/a_{}_{}_{}.pth'.format(configs.vm_types, configs.each_vm_type_num ,configs.arr_rate) ) 
                record = deepcopy(np.mean(meanFlowTimes))
            t1 = time.time()
            print('Vlidation at update-{}: mean_flowtime_deterministic: {:.3f}+/-{:.3f}\t record: {:.3f}\t time_elapsed: {:.3f}'.\
                  format(str(i_update+1), np.mean(meanFlowTimes), np.std(meanFlowTimes), record, (t1 - total1)/3600), flush=True)

    # np.save('./logs/cross_{}-{}-{}.npy'.format(configs.vm_types, configs.each_vm_type_num ,configs.arr_rate), cross_losses)
    # np.save('./logs/entropy_{}-{}-{}.npy'.format(configs.vm_types, configs.each_vm_type_num ,configs.arr_rate), ent_losses)

if __name__ == '__main__':
    total1 = time.time()
    main()
    total2 = time.time()
    print('>>>Overall Runtime is ', (total2 - total1)/3600, ' hours', flush=True)