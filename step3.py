from copy import deepcopy
import torch
import time, random, pickle
import numpy as np
from config.Params import configs
from policy.actor3 import PPO, BatchGraph, Memory, RolloutBuffer
from env.workflow_scheduling_v3.lib.poissonSampling import sample_poisson_shape
from env.workflow_scheduling_v3.simulator_wf import WFEnv
# import multiprocessing

device = torch.device(configs.device)
file_writing_a = './logs/actor_log.pkl'
file_writing_c = './logs/critic_log.pkl'
file_writing_train = './logs/train_log.txt'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)    


def collect_rollouts(env, policy, stop_numTimestep, memory, pre_state_list):   
    terminated = False
    # ep_rewards = 0
    batch_states = BatchGraph(configs.normalize)
    v_states = BatchGraph(configs.normalize)
    state_list = pre_state_list #env.reset()
    while True:
        with torch.no_grad():
            v_states.form_v_state(*state_list)    # Processed as v-fn input
            batch_states.wrapper(*state_list)   # processing (s,a)
            action, prob,_ = policy(state_wf = batch_states.wf_features,    
                                state_vm = batch_states.vm_features,                 
                                edge_index_wf = batch_states.wf_edges,    
                                edge_index_vm = batch_states.vm_edges,   
                                mask_wf = batch_states.wf_masks,
                                mask_vm = batch_states.vm_masks,
                                batch_wf = batch_states.wf_batchs,
                                batch_vm = batch_states.vm_batchs,
                                candidate_task_index = batch_states.candidate_taskID,
                                deterministic = False)

        state_list, reward, done = env.step(action.item())  
        memory.record( deepcopy(batch_states), deepcopy(action), deepcopy(reward), deepcopy(done), deepcopy(prob), deepcopy(v_states))

        if env.numTimestep == stop_numTimestep or done:  
            if done:
                terminated = True
            break

    return state_list, terminated

def compute_values(critic, bufferdata):
    start_idx = 0
    values = []
    while start_idx <  len(bufferdata):
        end_idx = min(start_idx + configs.batch_size, len(bufferdata))
        batch_states = BatchGraph(configs.normalize).batch_process(bufferdata[start_idx:end_idx]) 
        vals = critic(state_wf = batch_states.wf_features,         
                        edge_index_wf = batch_states.wf_edges,       
                        mask_wf = batch_states.wf_masks,
                        batch_wf = batch_states.wf_batchs,
                        candidate_task_index = batch_states.candidate_taskID,
                        deterministic = False) 
        values.extend(vals.tolist())
        start_idx += configs.batch_size    
    return values

def append_to_nested_lists(nested_lists, new_lists):
    for i in range(len(nested_lists)):
        nested_lists[i].append(new_lists[i])

def validation_Org(args, model, deterministic):
    envs = WFEnv(args.env_name, args, False)
    state_list = envs.reset()
    batch_states = BatchGraph(args.normalize)
    while True:
        with torch.no_grad():
            batch_states.wrapper(*state_list)  
            action, _, _ = model(state_wf = batch_states.wf_features,     
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
            return envs.all_flowTime

def compareBaselines(pre_array, array, targets):
    outputs = []
    if pre_array is None:
        non_zero_indices = np.nonzero(array)[0]
    else:
        non_zero_indices = np.where((array != 0) & (pre_array == 0))[0]
    for target in targets:
        a = array[non_zero_indices]
        b = target[non_zero_indices]
        outputs.append( 100* (np.mean(b) - np.mean(a)) / (np.mean(b) +1e-8))

    outputs.insert(0, np.mean(a))
    return outputs

def cumBaselines(pre_array, array, targets):
    outputs = []
    non_zero_indices = np.nonzero(array)[0]
    for target in targets:
        b = target[non_zero_indices]
        outputs.append( np.mean(b) )

    if pre_array is None:
        non_zero_indices = np.nonzero(array)[0]
    else:
        non_zero_indices = np.where((array != 0) & (pre_array == 0))[0]

    outputs.append(np.mean(array[non_zero_indices]))
    return outputs

def preriod_compare(array, targets):
    split_indices = [i for i in range(0,len(array),500)] 
    a = np.array([np.mean(array[split_indices[i]:split_indices[i+1]]) for i in range(len(split_indices)-1)])
    for k,target in enumerate(targets):
        b = np.array([np.mean(target[split_indices[i]:split_indices[i+1]]) for i in range(len(split_indices)-1)])
        mean_differences = b-a
        print('Segment mean difference with Baseline-'+ str(k) +' are ---->', mean_differences, flush=True)

def main():
    # Load dataset and Bulid env
    set_seed(2024)
    configs.valid_dataset = np.flip(np.load('./validation_data/validation_instance_2024.npy').reshape((1,1,-1)) )[:,:,:configs.wf_num]
    configs.GENindex = 0
    configs.indEVALindex = 0
    configs.arr_times = sample_poisson_shape(configs.arr_rate, configs.valid_dataset.shape)
    
    env = WFEnv(configs.env_name, configs, False)
    state_list = env.reset()

    # Load initial actor
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
    algos.actor.load_state_dict(torch.load('./validation_data/step2/actors/a_{}.pth'.format(configs.online_start_ac), map_location=torch.device(device), weights_only=True))

    # Calulate 
    t1 = time.time()
    baseline2 = validation_Org(configs, algos.actor, True)
    t2 = time.time()
    print('mean flowtime of all worfklows using Original Actor --> {:.4f} with time elapsed {:.4f} h'.format(np.mean(baseline2), (t2-t1)/3600), flush=True)

    set_seed(configs.algo_seed)
    if configs.separate_update==0:
        algos.optimizer = torch.optim.Adam([
                            {'params': algos.actor.parameters(), 'lr': configs.lr_a},
                            {'params': algos.critic.parameters(), 'lr': configs.lr_c}])  

    # Training loop
    log = []
    # aloss_log = [[], [], []]
    # closs_log = [[], []]
    pre_record = None
    algos.pre_grad_max = 0
    algos.entropy_count = 0
    algos.grad_count = 0
    buffer = Memory(configs.window_steps * 7)
    
    for i_update in range(configs.max_updates):
        # Rollout the env
        stop_numTimestep = configs.warmup_steps + configs.window_steps * i_update 
        state_list, terminated = collect_rollouts(env, algos.actor, stop_numTimestep, buffer, state_list)    

        # print compare
        record = deepcopy(env.all_flowTime)
        mean_ = np.mean(record[record != 0])
        compares = cumBaselines(pre_record, record, [ baseline2])
        if pre_record is None:  
            length_ = np.count_nonzero(record)         
        else:
            length_ = np.count_nonzero(record)-np.count_nonzero(pre_record)
        t1 = time.time()
        log.append([i_update, length_, mean_] + compares + [(t1-total1)/3600])
        print('Episode-{}: period_length: {}\t meanFlowTime: {:.4f}\t compare_OrgActor: {:.4f}\t period_meanFlowTime: {:.4f}\t time_elapsed: {:.4f}'.format(*log[-1]), flush=True)        
        # with open(file_writing_train, 'w') as f:
        #     for entry in log:
        #         f.write(str(entry) + '\n')

        if terminated is False: 

            ## Calculate returns
            period_returns = buffer.compute_returns_new()
            period_samples = RolloutBuffer(buffer, period_returns, period_returns, len(period_returns)) 

            for _ in range(configs.n_epochs):
                ## Update Critic
                new_values, closs = algos.train_critic(period_samples)
                if i_update > configs.warmup_critic:
                    ## Updata Actor
                    period_samples.update_advantages(new_values)
                    actor_samples = deepcopy(period_samples)
                    actor_samples.get_reversed_slice(configs.window_steps)
                    aloss = algos.train_actor(actor_samples) 
                else:
                    aloss = ([0], [0], [0])
                t1 = time.time()
                temp_log = [np.mean(aloss[0]), np.mean(aloss[1]), np.mean(aloss[2]), np.std(aloss[2]), algos.pre_grad_max,\
                                    np.mean(closs[0]), np.mean(closs[1]), np.std(closs[1]),(t1-total1)/3600]
                
                print('\t all_loss: {:.6f}\t entropy_loss: {:.6f}\t grad_changes: {:.6f}+/-{:.6f}\t grad_max: {:.6f}\t rmse_loss: {:.6f}\t mre_loss: {:.6f}+/-{:.6f}\t time_elapsed: {:.3f}'.format(\
                        *temp_log), flush=True) 

        else:
            break

        pre_record = deepcopy(record)
        if (i_update + 1) % configs.log_interval == 0:         
            torch.save(algos.actor.state_dict(), './logs/a-{}-{}-{}-{}.pth'.format(configs.online_start_ac, configs.vm_types, configs.each_vm_type_num, configs.arr_rate) )
            torch.save(algos.critic.state_dict(), './logs/c-{}-{}-{}-{}.pth'.format(configs.online_start_ac, configs.vm_types, configs.each_vm_type_num, configs.arr_rate) )
    
    if terminated is False: 
        state_list, terminated = collect_rollouts(env, algos.actor, 1e10, buffer, state_list)

        record = env.all_flowTime
        mean_ = np.mean(record)
        compares = cumBaselines(pre_record, record, [ baseline2])  
        length_ = np.count_nonzero(record)-np.count_nonzero(pre_record)
        t1 = time.time()
        log.append([i_update+1, length_, mean_] + compares + [(t1-total1)/3600])
        print('Episode-{}: period_length: {}\t meanFlowTime: {:.4f}\t compare_OrgActor: {:.4f}\t period_meanFlowTime: {:.4f}\t time_elapsed: {:.4f}'.format(*log[-1]), flush=True)        
        
    torch.save(algos.actor.state_dict(), './logs/a-{}-{}-{}-{}.pth'.format(configs.online_start_ac, configs.vm_types, configs.each_vm_type_num, configs.arr_rate) )
    torch.save(algos.critic.state_dict(), './logs/c-{}-{}-{}-{}.pth'.format(configs.online_start_ac, configs.vm_types, configs.each_vm_type_num, configs.arr_rate) )
    t1 = time.time()

    print('trigger entropy {} times, trigger gradient_control {} times'.format(algos.entropy_count, algos.grad_count), flush=True)
    print('>>> Total env steps is ', env.numTimestep, flush=True)
    print('mean flowtime of all worfklows using Online Actor --> {:.4f} with time elapsed {:.4f} h'.format(np.mean(env.all_flowTime), (t1-total1)/3600), flush=True)

    set_seed(2024)
    t0 = time.time()
    baseline3 = validation_Org(configs, algos.actor, True)
    t1 = time.time()
    print('mean flowtime of all worfklows using Final Actor --> {:.4f} with time elapsed {:.4f} h'.format(np.mean(baseline3), (t1-t0)/3600), flush=True)

    np.save( './logs/determine_log.npy', np.array([ baseline2, baseline3, env.all_flowTime]))
    preriod_compare(env.all_flowTime, [ baseline2, baseline3])

if __name__ == '__main__':
    total1 = time.time()
    main()
    total2 = time.time()
    print('>>> Overall Runtime is ', (total2 - total1)/3600, ' hours', flush=True)