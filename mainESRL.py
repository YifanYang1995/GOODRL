from copy import deepcopy
import torch
import time, random
import numpy as np
from config.Params import configs
from env.workflow_scheduling_v3.lib.poissonSampling import sample_poisson_shape
from policy.es_rl import ESOpenAI
from joblib import Parallel, delayed
from policy.actor3 import BatchGraph
from env.workflow_scheduling_v3.simulator_wf import WFEnv

device = torch.device(configs.device)
# file_writing_a = './logs/actor_log_' + str(configs.epochs_c) + '_' + str(configs.lr_c) + '_' + str(configs.window_steps) + '.npy'
# file_writing_c = './logs/critic_log_' + str(configs.epochs_c) + '_' + str(configs.lr_c) + '_' + str(configs.window_steps) + '.npy'
# file_writing_ = './logs/actor_log_' + str(configs.epochs_c) + '_' + str(configs.lr_c) + '_' + str(configs.window_steps) + '.pkl'
# file_writing_g = './logs/grad_log_' + str(configs.epochs_c) + '_' + str(configs.lr_c) + '_' + str(configs.window_steps) + '.pkl'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)    

def validation_Org(i1, i2, args, model, trainset):
    # 验证在original_actor上的结果
    # set_seed(configs.env_seed)
    en_loss = []
    args.GENindex = i1
    args.indEVALindex = i2
    envs = WFEnv(args.env_name, args, trainset)
    state_list = envs.reset()
    batch_states = BatchGraph(args.normalize)
    while True:
        with torch.no_grad():
            batch_states.wrapper(*state_list)   # 处理(s,a)
            action, _, entropy = model(state_wf = batch_states.wf_features,      ## (batch_szie^2, 2)
                                state_vm = batch_states.vm_features,                 
                                edge_index_wf = batch_states.wf_edges,    
                                edge_index_vm = batch_states.vm_edges,   
                                mask_wf = batch_states.wf_masks,
                                mask_vm = batch_states.vm_masks,
                                batch_wf = batch_states.wf_batchs,
                                batch_vm = batch_states.vm_batchs,
                                candidate_task_index = batch_states.candidate_taskID,
                                deterministic = True)
        en_loss.append(entropy.item())
        state_list, _, done = envs.step(action.item())  
        if done:  
            return np.mean(envs.all_flowTime), np.mean(en_loss)  

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

def main():
    # Load dataset
    record = 1e10
    wf_types=4
    set_seed(configs.env_seed)

    configs.valid_dataset = np.load('./validation_data/validation_instance_2024.npy').reshape((1,-1, configs.wf_num)) [:, :(configs.valid_num + configs.num_envs)]
    configs.GENindex = 0
    configs.indEVALindex = 0
    configs.arr_times = sample_poisson_shape(configs.arr_rate, configs.valid_dataset.shape)  
    configs.train_dataset = np.random.randint(0,wf_types,(configs.es_gen_num+1, configs.eval_num, configs.wf_num ))
    configs.arr_times_train = sample_poisson_shape(configs.arr_rate, configs.train_dataset.shape)

    meanFlowTimes = Parallel(n_jobs=-1)(delayed(validation_H)( t, configs ) for t in range(configs.valid_num)) 
    t1 = time.time()
    print('Vlidation at HEFT: mean_flowtime_deterministic: {:.6f}+/-{:.6f}\t time_elapsed: {:.6f}'.\
            format(np.mean(meanFlowTimes), np.std(meanFlowTimes), (t1 - total1)/3600), flush=True)    

    # Build policy
    set_seed(configs.algo_seed)
    params=  torch.load('./validation_data/ESRL/a_{}_{}_{}.pth'.format(configs.vm_types, configs.each_vm_type_num, configs.arr_rate),\
                       weights_only=True, map_location=torch.device(device)) # after imitation learning
    algos = ESOpenAI(configs.policy_name,
                     input_dim_wf = configs.input_dim_wf,
                     input_dim_vm= configs.input_dim_vm,           
                     hidden_dim= configs.hidden_dim,
                     gnn_layers= configs.gnn_layers,
                     mlp_layers= configs.mlp_layers,  
                     para = params,                                                                             
                    )

    # Training loop
    log = []
    valid_log = []

    population = algos.init_population()
    for i_update in range(configs.es_gen_num+1):
        results = Parallel(n_jobs=-1)(delayed(validation_Org)( i_update, 0, configs, ind, True ) for ind in population)
        results = np.array(results)

        if i_update % configs.log_interval == 0:
            curr_best_policy = algos.get_elite_model()
            valids = Parallel(n_jobs=-1)(delayed(validation_Org)( 0, t, configs, curr_best_policy, False ) for t in range(configs.valid_num))  
            valids = np.array(valids)
            meanFlowTimes,meanEntropies = np.mean(valids,axis=0)
            stdFlowTimes,stdEntropies = np.std(valids,axis=0)
            if meanFlowTimes < record:
                record = deepcopy(meanFlowTimes)    
                torch.save(curr_best_policy, './logs/a_{}_{}_{}.pth'.format(configs.vm_types, configs.each_vm_type_num, configs.arr_rate) )
            t1 = time.time()
            valid_log.append([i_update, meanFlowTimes, stdFlowTimes, meanEntropies, stdEntropies ,record, (t1 - total1)/3600])
            print('Vlidation at update-{}: mean_flowtime_deterministic: {:.6f}+/-{:.6f}\t mean_Entropy: {:.6f}+/-{:.6f}\t record: {:.6f}\t time_elapsed: {:.6f}'.\
                    format(*valid_log[-1]), flush=True)

        population, _, grad = algos.next_population(results[:,0]) # update parameters
        t1 = time.time()
        log.append([i_update, np.mean(results[:,0]), np.std(results[:,0]), np.min(results[:,0]), np.mean(results[:,1]),np.std(results[:,1]), grad, (t1-total1)/3600])
        print('Episode-{}: avg: {:.6f}\t std: {:.6f}\t min: {:.6f}\t e_loss: {:.6f}+/-{:.6f}\t grad: {:.6f}\t time_elapsed: {:.6f}'.format(*log[-1]), flush=True)  

if __name__ == '__main__':
    total1 = time.time()
    main()
    total2 = time.time()
    print('>>>Overall Runtime is ', (total2 - total1)/3600, ' hours', flush=True)