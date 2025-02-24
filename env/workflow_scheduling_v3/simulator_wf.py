from env.workflow_scheduling_v3.lib.cloud_env_maxPktNum import cloud_simulator
import numpy as np
import env.workflow_scheduling_v3.lib.dataset as dataset



class WFEnv(cloud_simulator):
    def __init__(self, name, args, trainModel=True):

        if hasattr(args, 'train_dataset'):
            train_dataset = args.train_dataset       
        else:
            # Create a data set 
            wf_types=len(dataset.dataset_dict[args.wf_size])
            if args.generateWay == 'fixed':
                train_dataset = np.random.randint(0,wf_types,(1, args.num_envs, args.wf_num))
                train_dataset = np.array([list(train_dataset[0]) for _ in range(args.max_updates+1)])
                train_dataset = train_dataset.astype(np.int64)
            else:   # by rotation
                train_dataset = np.random.randint(0,wf_types,(args.max_updates+1, args.num_envs, args.wf_num ))
                train_dataset = train_dataset.astype(np.int64)  

        if hasattr(args, 'valid_dataset'):
            valid_dataset = args.valid_dataset         

        # Setup
        if trainModel == True:
            config = {"traf_type": args.traf_type, "seed": args.env_seed, "arr_rate": args.arr_rate, "envid": 0, "vm_types":args.vm_types,
                  "wf_size": args.wf_size, "wf_num": args.wf_num, "trainSet": train_dataset, "each_vm_type_num":args.each_vm_type_num} 
        else: 
            config = {"traf_type": args.traf_type, "seed": args.env_seed, "arr_rate": args.arr_rate, "envid": 0, "vm_types":args.vm_types,
                  "wf_size": args.wf_size, "wf_num": args.wf_num, "trainSet": valid_dataset, "each_vm_type_num":args.each_vm_type_num}

        super().__init__(config)
        self.name = name
        self.args = args
        self.train_or_test = trainModel
        

    def reset(self):
        super().reset(self.args.GENindex, self.args.indEVALindex)
        self.step_curr = self.numTimestep
        if self.args.require_estimated_features in [0,1]:
            states = self.state_info_construct1()
        else:
            states = self.state_info_construct2()
        return states

    def step(self, action):

        reward, done = super().step(action)
        if done:
            state_list = self.episode_info
        else:
            if self.args.require_estimated_features in [0,1]:
                state_list = self.state_info_construct1()
            else:
                state_list = self.state_info_construct2()

        return state_list, reward, done, 


    def resetGP(self):
        super().reset(self.args.GENindex, self.args.indEVALindex)
        states = self.gp_feature_construct()
        return states
    def stepGP(self, action):
        reward, done = super().step(action)
        if done:
            state_list = self.episode_info
        else:
            state_list = self.gp_feature_construct()      
        return state_list, reward, done  



    def resetES(self):
        super().reset(self.args.GENindex, self.args.indEVALindex)
        states = self.esrl_feature_construct()
        return states
    def stepES(self, action):
        reward, done = super().step(action)
        if done:
            state_list = self.episode_info
        else:
            state_list = self.esrl_feature_construct()      
        return state_list, reward, done  


    def get_agent_ids(self):
        return ["0"]

    def close(self):
        super().close()
