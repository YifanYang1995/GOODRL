import multiprocessing
from config.Params import configs
from env.workflow_scheduling_v3.simulator_wf import WFEnv
from env.workflow_scheduling_v3.lib.poissonSampling import sample_poisson_shape

import pandas as pd
import numpy as np
import operator,random,time
from functools import partial

from deap import base
from deap import creator
from deap import tools
from deap import gp


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

pset1 = gp.PrimitiveSet("main1", 6)  ##The second arity is input numbers, == Terminal
pset1.addPrimitive(np.maximum, 2)#, name='max')
pset1.addPrimitive(np.minimum, 2)#, name='min')
pset1.addPrimitive(np.add, 2)#, name='+')
pset1.addPrimitive(np.subtract, 2)#, name='-')
pset1.addPrimitive(np.multiply, 2)#, name='*')
pset1.addPrimitive(protectedDiv, 2, name='div')
pset1.addEphemeralConstant("rand101", partial(random.randint, -1, 1))

### terminals rename
pset1.renameArguments(ARG0='TS')  # task_size of a task
pset1.renameArguments(ARG1='RW')  # remain workload in the workflow
pset1.renameArguments(ARG2='ET')  # execute_time_period of a task on this VM
pset1.renameArguments(ARG3='FT')  # finish_time = start_time + ET
pset1.renameArguments(ARG4='CU')  # compute unit of a VM 
pset1.renameArguments(ARG5='UL')  # utilization of a VM 

# create container
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

## create methods
toolbox = base.Toolbox()
toolbox.register("expr1", gp.genHalfAndHalf, pset=pset1, min_=2, max_=6)  #
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset1)


def gptreePolicy(tree, obs):

    priorities = []
    rule = toolbox.compile(expr = tree)

    if configs.normalize:
        obs = obs / np.array(configs.normalize_features[1:])
        
    priorities = rule(*obs.T)
    if isinstance(priorities, int) or priorities.ndim ==0:     # Featureless rule, i.e., random
        priorities = np.full((len(obs),), 1, dtype=np.float64)
    min_indices = np.where(priorities == np.min(priorities))[0]

    return np.random.choice(min_indices)

def collect_rollouts(items):    ## evalution on one problem instance
    
    args,tree, k, trainortest = items
    if trainortest:
        args.GENindex = k
        args.indEVALindex = 0
    else:
        args.GENindex = 0
        args.indEVALindex = k        

    env = WFEnv(args.env_name, args, trainortest)
    
    state_list = env.resetGP()
    ep_rewards = 0
    while True:
        action = gptreePolicy(tree, state_list)
        state_list, reward, done = env.stepGP(action)   ## 比env.rest()多了reward, done
        ep_rewards +=reward
        if done:  
            # print(env.numTimestep)
            return np.mean(env.all_flowTime), #-ep_rewards, # makespan

toolbox.register("evaluate", collect_rollouts)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("selectElitism", tools.selBest, k=configs.elite_num)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset1) 

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

# toolbox.register("validate", validation)

def varOr(population, toolbox, cxpb, mutpb): # population already be processed by the ternament selection
    offspring = [toolbox.clone(ind) for ind in population] 
    crossover_index = []
    mutation_index = []
    reprod_index=[]
    for i in range(len(offspring)):
        r = random.random()
        if r < mutpb:
            mutation_index.append(i)
        elif r < cxpb + mutpb :
            crossover_index.append(i)
        else:
            reprod_index.append(i)     
    # Reproduction
    for i in range(len(reprod_index)):                   
        offspring[reprod_index[i]], = [toolbox.clone(offspring[reprod_index[i]])]
        del offspring[reprod_index[i]].fitness.values            
    # Crossover
    for i in range(1, len(crossover_index), 2):
        offspring[crossover_index[i-1]], offspring[crossover_index[i]] = toolbox.mate(offspring[crossover_index[i-1]], offspring[crossover_index[i]])
        del offspring[crossover_index[i-1]].fitness.values, offspring[crossover_index[i]].fitness.values
    # Mutation
    for i in range(len(mutation_index)): 
        offspring[mutation_index[i]], = toolbox.mutate(offspring[mutation_index[i]]) 
        del offspring[mutation_index[i]].fitness.values   
    return offspring


def main():

    # Load dataset
    max_fitness = 1e10
    wf_types=4  
    set_seed(configs.env_seed)
    configs.valid_dataset = np.load('./validation_data/validation_instance_2024.npy').reshape((1,-1, configs.wf_num)) [:, :(configs.valid_num + configs.num_envs)]
    configs.GENindex = 0
    configs.indEVALindex = 0
    configs.arr_times = sample_poisson_shape(configs.arr_rate, configs.valid_dataset.shape)
    configs.train_dataset = np.random.randint(0,wf_types,(configs.gen_num+1, configs.eval_num, configs.wf_num))    
    configs.arr_times_train = sample_poisson_shape(configs.arr_rate, configs.train_dataset.shape)

    # Bulid GP trees
    set_seed(configs.algo_seed)
    population = toolbox.population(n=configs.pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    
    pop_log, validation_log, bestTree_log = [],[],[]

    # Initial generation
    for ind in population: 
        ind.archive_results = [] 
    for i in range(configs.eval_num):
        invalid_ind = [(configs, ind, 0, True) for ind in population]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(population, fitnesses):
            ind.archive_results.append(fit[0])
    for ind in population: 
        ind.fitness.values = np.mean(ind.archive_results), #np.std(ind.archive_results))

    # Elistism in the first generation
    elites_in_pop = toolbox.selectElitism(population)
    elites = [toolbox.clone(ind) for ind in elites_in_pop]
    bestTree_log.append(toolbox.clone(elites[0])) 

    # Record
    # print('*************** Generation {} ***************'.format(0), flush=True)
    hof.update(population)
    record = stats.compile(population) if stats else {}
    toprint = {k: v for k, v in record.items() if not isinstance(v, dict)}
    t1 = time.time()
    pop_log.append([0] + list(toprint.values())+ [(t1- total1)/3600 ])
    # file_writing_obj = open('./logs/log_' + str(configs.wf_size) + '_' + str(configs.wf_num) + '.txt', 'w')
    # file_writing_obj.write(str(pop_log))
    # if pop_log[-1][-2] < max_fitness:
    #     pd.to_pickle(bestTree_log[-1], './logs/bestTree.pkl')
    #     max_fitness = pop_log[-1][-2]    

    print('Generation-{} --> avg: {:.3f}\t std: {:.3f}\t  min: {:.3f}\t time_elapsed: {:.3f}'.format(*pop_log[-1]), flush=True)
    # Validation
    # results = toolbox.validate(bestTree_log[-1], configs)
    invalid_ind = [(configs, bestTree_log[-1], k, False) for k in range(configs.valid_num)]
    results = toolbox.map(toolbox.evaluate, invalid_ind)
    results = [item[0] for item in results]
    validation_log.append([0, np.mean(results), np.std(results)])
    # file_writing_obj1 = open('./logs/vali_' + str(configs.wf_size) + '_' + str(configs.wf_num) + '.txt', 'w')
    # file_writing_obj1.write(str(validation_log))
    t1 = time.time()
    print('Validation result ---->\t mean_flowtime: {:.3f} +/- {:.3f}\t time_elapsed: {:.3f}'.\
              format(validation_log[-1][1], validation_log[-1][2], (t1- total1)/3600 ), flush=True) 

    for gen in range(1, configs.gen_num + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population)-len(elites))    

        # Vary the pool of individuals
        offspring = varOr(offspring, toolbox, configs.cross_rate, configs.mutate_rate)
        offspring[0:0] = elites

        # Evaluate the individuals with an invalid fitness
        # configs.GENindex = gen 
        for ind in offspring: 
            ind.archive_results = [] 
        for i in range(configs.eval_num):
            invalid_ind = [(configs, ind, gen, True) for ind in offspring] # np.random.randint(configs.num_envs)
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(offspring, fitnesses):
                ind.archive_results.append(fit[0])
        for ind in offspring: 
            ind.fitness.values = np.mean(ind.archive_results),

        population[:] = offspring

        # Elistism
        elites_in_pop = toolbox.selectElitism(population)
        elites = [toolbox.clone(ind) for ind in elites_in_pop]
        bestTree_log.append(toolbox.clone(elites[0]))         

        # Update infos
        # print('*************** Generation {} ***************'.format(gen), flush=True)
        hof.update(population)
        record = stats.compile(population) if stats else {}
        toprint = {k: v for k, v in record.items() if not isinstance(v, dict)}        
        t1 = time.time()
        pop_log.append([gen] + list(toprint.values())+ [(t1- total1)/3600 ])
        # file_writing_obj = open('./logs/' + 'log_' + str(configs.wf_size) + '_' + str(configs.wf_num) + '.txt', 'w')
        # file_writing_obj.write(str(pop_log))
        print('Generation-{} --> avg: {:.3f}\t std: {:.3f}\t  min: {:.3f}\t time_elapsed: {:.3f}'.format(*pop_log[-1]), flush=True)

        # results = toolbox.validate(bestTree_log[-1], configs)
        invalid_ind = [(configs, bestTree_log[-1], k, False) for k in range(configs.valid_num)]
        results = toolbox.map(toolbox.evaluate, invalid_ind)
        results = [item[0] for item in results]
        validation_log.append([gen, np.mean(results), np.std(results)])   
        
        # file_writing_obj1 = open('./logs/vali_' + str(configs.wf_size) + '_' + str(configs.wf_num) + '.txt', 'w')
        # file_writing_obj1.write(str(validation_log))
        t1 = time.time()
        print('Validation result ---->\t mean_flowtime: {:.3f} +/- {:.3f}\t time_elapsed: {:.3f}'.\
              format(validation_log[-1][1], validation_log[-1][2], (t1- total1)/3600 ), flush=True) 
        # Save the global best tree
        if validation_log[-1][1] < max_fitness:
            pd.to_pickle(bestTree_log[-1], './logs/bestTree_{}_{}_{}.pkl'.format(configs.vm_types, configs.each_vm_type_num, configs.arr_rate))
            max_fitness = validation_log[-1][1]

    pd.to_pickle(hof[0], './logs/bestTree_hof.pkl')

if __name__ == '__main__':
    total1 = time.time()
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)    
    main()
    pool.close()
    total2 = time.time()
    print('>>>Overall Runtime is ', (total2 - total1)/3600, ' hours', flush=True)