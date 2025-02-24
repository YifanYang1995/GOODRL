import numpy as np
import pandas as pd
import csv
import math
import os, sys, inspect, random, copy
import gym
from config.Params import configs
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from env.workflow_scheduling_v3.lib.stats import Stats
from env.workflow_scheduling_v3.lib.poissonSampling import one_sample_poisson
from env.workflow_scheduling_v3.lib.vm import VM
from env.workflow_scheduling_v3.lib.workflow import Workflow
from env.workflow_scheduling_v3.lib.simqueue import SimQueue
from env.workflow_scheduling_v3.lib.simsetting import Setting
from env.workflow_scheduling_v3.lib.processDAG import compute_initial_earliest_finish_times, update_node_weight_and_compute_finish_times

vmidRange = 10000

def ensure_dir_exist(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_csv_header(file, header):
    ensure_dir_exist(file)
    with open(file, 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(header)

def write_csv_data(file, data):
    ensure_dir_exist(file)
    with open(file, 'a', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(data)


class cloud_simulator(object):

    def __init__(self, args):

        self.set = Setting(args)
        self.baseHEFT = None
        self.trainSet = args["trainSet"]
        self.GENindex = None
        self.indEVALindex = None       
        self.TaskRule = None                # input th task selection rule here, if has
        # self.validSet = args["Validation"]

        if self.set.is_wf_trace_record:
            self.df = {}
            __location__ = os.getcwd() + '\Saved_Results'
            self.pkt_trace_file = os.path.join(__location__, r'allocation_trace_%s_seed%s_arr%s.csv' % (args["algo"],  args["seed"], args["arrival rate"]))
            write_csv_header(self.pkt_trace_file, ['Workflow ID', 'Workflow Pattern', 'Workflow Arrival Time', 'Workflow Finish Time', 'Workflow Deadline', 'Workflow Deadline Penalty',
                                                   'Task Index', 'Task Size', 'Task Execution Time', 'Task Ready Time', 'Task Start Time', 'Task Finish Time',
                                                   'VM ID', 'VM speed', 'Price', 'VM Rent Start Time', 'VM Rent End Time', 'VM Pending Index' ]) # 6 + 6 + 6 columns     

        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(6 + self.set.history_len,))
        self.action_space = gym.spaces.Discrete(n=100)  # n is a placeholder
        


    def close(self):
        print("Environment id %s is closed" % (self.set.envid))


    def _init(self):

        # Set record 12 features
        columns = ['Workflow ID', 'Task Index', 'Task Size', 'Wd2End', 'RemainWd in Wf', 'Task Schedule status', 'Task Ready Time',\
                   'Task Waiting Time', 'Task Execute Time', 'Task Finish Time', 'VM ID', 'VM speed', 'VM Process Index', 'VM Utilization']
        self.df = pd.DataFrame(columns=columns)        
        # self.all_flowTime = []
        # Parameter
        self.appSubDeadline = {}  # { app: {task: task_sub_deadline} } used as a state feature
        self.usr_queues = []            # [usr1:[workflows, ...], usr2:[workflows, ...]], e.g., user1 stores 30 workflows
        self.finishWf_queues = []
        self.activeWf_queues = []       # all complete and on-going workflows
        self.vm_queues = []             # [VM1, VM2, ...] each VM is a class
        self.vm_queues_id = []          # the vmIndex of each VM in self.vm_queues
        self.vm_queues_cpu = []
        self.vm_queues_utl = np.zeros((self.set.vmNum))
        # self.vm_queues_rentEndTime = []
        self.usrNum = self.set.usrNum   ## useless for one usr
        self.dcNum = self.set.dcNum     ## useless for one cloud
        self.wfTypes = self.set.wfTypes
        self.wfNum= self.set.WorkflowNum
        self.vmNum = self.set.vmNum
        self.vmEachIntNum = self.set.vmEachIntNum       
        self.VMtypeNum = len(self.set.dataset.vmVCPU) ## number of VM types
        self.numTimestep = 0            # indicate how many timesteps have been processed
        self.completedWF = 0
        # self.VMRemainingTime = {}       # {vmid1:time, vmid2:time} Not considering the idle gap time
        # self.VMRemainAvaiTime = {}      # reamin available time  = leased time period - vm_total_execute_time
        self.VMrentInfos = {}           # {VMid: [rent start time, rent end time]}
        self.notNormalized_arr_hist = np.zeros((self.usrNum, self.wfTypes, self.set.history_len)) 
        # self.VMcost = 0
        # self.SLApenalty = 0
        self.wfIndex = 0
        self.vmIndex = 10000            # labeled begin with 10000
        self.usrcurrentTime = np.zeros(self.usrNum)  # Used to record the current moment of the user
        self.dcCurrentTime = np.zeros(self.dcNum) 
        self.remainWrfNum = 0           # Record the number of packets remained in VMs
        # self.missDeadlineNum = 0
        # self.VMrentHours = 0  
        self.VMexecHours = 0        # The sum of processing time on all VMs
        self.VMidleTime = {}            ## Record the sum of the VM's current idle time to calculate VM utilization, += max(machine time, current time)      

        # IMPORTANT: used to get the ready task for the next time step
        self.firstvmWrfLeaveTime = []   # Record the current timestamp on each VM
        self.firstusrWrfGenTime = np.zeros(self.usrNum)  # Arrival time of the first inactive workflow in each user's workflow set
        self.vmLatestTime = {}

        self.uselessAllocation = 0
        # self.VMtobeRemove = None
        self.all_flowTime = np.zeros((self.wfNum))
        self.usr_respTime = np.zeros((self.usrNum, self.wfNum)) 
        self.usr_received_wrfNum = np.zeros((self.usrNum, self.wfTypes)) 
        self.usr_sent_pktNum = np.zeros((self.usrNum, self.dcNum))

        self.totalTaskSize = 0
        self.cumulative_reward_M = 0
        self.returns_M = 0
        self.maxVms = 0

        # upload all workflows with their arrival time to the 'self.firstusrWrfGenTime'
        for i in range(self.usrNum):
            self.usr_queues.append(SimQueue())
            if self.train_or_test:
                workflowsIDs = self.args.train_dataset[self.GENindex][self.indEVALindex]   
                arrTimes = self.args.arr_times_train[self.GENindex][self.indEVALindex] 
            else:
                workflowsIDs = self.args.valid_dataset[self.GENindex][self.indEVALindex]   
                arrTimes = self.args.arr_times[self.GENindex][self.indEVALindex] 
            for wfID,arrT in zip(workflowsIDs,arrTimes):
                self.workflow_generator(i, wfID, arrT)
            self.firstusrWrfGenTime[i] = self.usr_queues[i].getFirstWfEnqueueTime() 

            for vmID in range(len(self.set.dataset.vmVCPU)):
                self.vm_generator(0, vmID)

        self.nextUsr, self.nextTimeStep = self.get_nextWrfFromUsr() 
        self.PrenextTimeStep = self.nextTimeStep
        self.nextisUsr = True
        self.nextWrf, self.finishTask = self.usr_queues[self.nextUsr].getFirstWf() # obtain the root task of the first workflow in the self.nextUsr
        temp = self.nextWrf.get_allnextTask(self.finishTask)   # Get all real successor tasks of the virtual workflow root task
                
        self.dispatchParallelTaskNum = 0
        self.nextTask = temp[self.dispatchParallelTaskNum]
        if len(temp) > 1:  # the next task has parallel successor tasks
            self.isDequeue = False
            self.isNextTaskParallel = True
        else:
            self.isDequeue = True  # decide whether the nextWrf should be dequeued
            self.isNextTaskParallel = False

        self.check_add_newWf2df()
        self.stat = Stats(self.set)


    def check_add_newWf2df(self):
        if self.finishTask==None and self.nextWrf.appArivalIndex not in self.df['Workflow ID'].values:
            self.activeWf_queues.append(self.nextWrf)
            if configs.require_estimated_features in [1,2]:
                temp_earliest_finish_times = compute_initial_earliest_finish_times(self.nextWrf.app, self.nextWrf.generateTime)
            for node in self.nextWrf.app.nodes:
                if configs.require_estimated_features in [1,2]: # True
                    new_row = {'Workflow ID':self.nextWrf.appArivalIndex, 
                            'Task Index': node,
                                'Task Size':self.nextWrf.get_taskSize(node), 
                                'Wd2End':self.nextWrf.get_allsucc2end(node),
                                'RemainWd in Wf':self.nextWrf.remainWorkload, 
                                'Task Schedule status':3, 
                                'Task Ready Time': None, 'Task Execute Time': self.nextWrf.app.nodes[node]['estimatedET'], 
                                'Task Waiting Time':None, 'Task Finish Time': temp_earliest_finish_times[node],
                                'VM ID':None, 'VM speed': self.set.dataset.meanCPU, 'VM Process Index':None,\
                                'VM Utilization': None} #np.mean(self.vm_queues_utl)}
                else:
                    new_row = {'Workflow ID':self.nextWrf.appArivalIndex, 
                            'Task Index': node,
                                'Task Size':self.nextWrf.get_taskSize(node), 
                                'Wd2End':self.nextWrf.get_allsucc2end(node),
                                'RemainWd in Wf':self.nextWrf.remainWorkload, 
                                'Task Schedule status':3, 
                                'Task Ready Time': None, 'Task Execute Time': None, 
                                'Task Waiting Time':None, 'Task Finish Time': None,
                                'VM ID':None, 'VM speed': None, 'VM Process Index':None,\
                                'VM Utilization': None}                    
                # self.df = pd.concat([self.df, pd.DataFrame([new_row], columns=self.df.columns)], ignore_index=True)
                self.df.loc[len(self.df)] = new_row
                
                # 'Task Size':self.nextWrf.get_taskSize(self.nextTask), 'Wd2End':self.nextWrf.get_path2end(self.nextTask),\
            self.update_ready_tasks(self.nextWrf, self.finishTask)

    def update_ready_tasks(self, app, task):
        temp = app.get_allnextTask(task)
        for node in temp:   ## ready tasks
            parentTasks = app.get_allpreviousTask(node)
            if len(parentTasks) == len(app.completeTaskSet(parentTasks)):
                self.df.loc[(self.df['Workflow ID'] == app.appArivalIndex) & (self.df['Task Index'] == node), 
                        ['Task Schedule status', 'Task Ready Time', 'RemainWd in Wf']] = [2, self.nextTimeStep, app.remainWorkload] 

    # Generate one workflow at one time
    def workflow_generator(self, usr, wfID, nextarrT):

        wfDAG = self.set.dataset.wset[wfID]
        # nextArrivalTime = one_sample_poisson(
                            # self.set.get_individual_arrival_rate(self.usrcurrentTime[usr], usr, wfID), 
                            # self.usrcurrentTime[usr])

        self.remainWrfNum += 1
        # add workflow deadline to the workflow
        Wf = Workflow(self.usrcurrentTime[usr], wfDAG, wfID, usr, self.set.dataset.wsetSlowestT[wfID],\
                      self.set.dataset.wsetTotProcessTime[wfID], self.wfIndex) # ,self.set.dueTimeCoef[usr, appID])

        self.usr_queues[usr].enqueue(Wf, self.usrcurrentTime[usr], None, usr, 0) # None means that workflow has not started yet
        self.usrcurrentTime[usr] = nextarrT
        self.wfNum-= 1
        self.wfIndex +=1


    def vm_generator(self, dcID, vmTypeID):

        for _ in range(self.set.vmEachIntNum):
            self.vmIndex+=1
            newVM = VM(self.vmIndex, self.set.dataset.vmVCPU[vmTypeID], dcID, self.set.dataset.datacenter[dcID][0], self.dcCurrentTime[dcID], self.TaskRule)
            self.vm_queues.append(newVM)
            self.firstvmWrfLeaveTime.append(newVM.get_firstTaskDequeueTime()) #new VM is math.inf
            self.vm_queues_id.append(self.vmIndex)
            self.vm_queues_cpu.append(self.set.dataset.vmVCPU[vmTypeID]) 
            self.VMrentInfos[self.vmIndex] = [self.vmIndex, self.set.dataset.vmVCPU[vmTypeID],  self.set.dataset.vmPrice[self.set.dataset.vmVCPU[vmTypeID]], 
                                        self.dcCurrentTime[dcID]]     


    def reset(self, seed1=0, seed2=0):
        random.seed(configs.env_seed)
        np.random.seed(configs.env_seed)
        self.GENindex = seed1
        self.indEVALindex = seed2   ## each individual is evaluated on only one problem instance
        self._init()
        self.PrenextWrf , self.PrenextTask = self.nextWrf, self.nextTask
        self.update_VMrentInfos()
        self.PreMakespan =  max(self.vmLatestTime.values())

    def input_task_rule(self, rule):
        self.TaskRule = rule

    def generate_vmid(self):
        vmIndex = np.random.randint(vmidRange, size=1)[0]
        while vmIndex in self.VMRemainingTime:
            vmIndex = np.random.randint(vmidRange, size=1)[0]
        return vmIndex

    def get_nextWrfFromUsr(self):       # Select the User with the smallest timestamp
        usrInd = np.argmin(self.firstusrWrfGenTime)
        firstWfTime = self.firstusrWrfGenTime[usrInd]
        return usrInd, firstWfTime     # Returns the user and arrival time of the minimum arrival time of the workflow in the current User queue.

    def get_nextWrfFromVM(self):        # Select the machine with the smallest timestamp
        if len(self.firstvmWrfLeaveTime) > 0:
            vmInd = np.argmin(self.firstvmWrfLeaveTime)
            firstPktTime = self.firstvmWrfLeaveTime[vmInd]
            return vmInd, firstPktTime  # Returns vm-id and the minimum end time of the current VM
        else:
            return None, math.inf

    def get_nextTimeStep(self):

        self.PrenextUsr, self.PrenextTimeStep = self.nextUsr, self.nextTimeStep
        tempnextloc, tempnextTimeStep = self.get_nextWrfFromUsr()  
        tempnextloc1, tempnextTimeStep1 = self.get_nextWrfFromVM() 
        if tempnextTimeStep > tempnextTimeStep1:  # task ready time > VM minimum time
            self.nextUsr, self.nextTimeStep = tempnextloc1, tempnextTimeStep1  
                                        # The next step is to process the VM and update it to the timestep of the VM.
            self.nextisUsr = False
            self.nextWrf, self.finishTask = self.vm_queues[self.nextUsr].get_firstDequeueTask() # Only returns time, does not process task
        else:  # tempnextTimeStep <= tempnextTimeStep1
            if tempnextTimeStep == math.inf:   ## tempnextTimeStep：when self.usr_queues.queue is []
                self.nextTimeStep = None       ## tempnextTimeStep1：when self.firstvmWrfLeaveTime is []
                self.nextUsr = None
                self.nextWrf = None
                self.nextisUsr = True
            else:
                self.nextUsr, self.nextTimeStep = tempnextloc, tempnextTimeStep # Next step is to process user & Update to user's timeStep
                self.nextisUsr = True    # Activate new Workflow from Usr_queue
                self.nextWrf, self.finishTask = self.usr_queues[self.nextUsr].getFirstWf() # The current first task in the selected user

        self.check_add_newWf2df()

    def update_VMrentInfos(self):

        for i, vm in zip(self.VMrentInfos, self.vm_queues):
            self.VMrentInfos[i][-1] = vm.currentTimeStep
            self.vmLatestTime[i] = vm.currentTimeStep + vm. pendingTaskTime
        for i, vm in enumerate(self.vm_queues):
            self.vm_queues_utl[i] = vm.get_vmUtilization(self.nextTimeStep)


    def step(self, action):

        complete1Wf = False
        currentReward = 0

        selectedVMind = action

        # reward_M = 0
        self.PrenextUsr, self.PrenextTimeStep = self.nextUsr, self.nextTimeStep      
        self.numTimestep += 1  ## useless for GP

        # dispatch nextTask to selectedVM and update the wrfLeaveTime on selectedVM 
        parentTasks = self.PrenextWrf.get_allpreviousTask(self.PrenextTask)
        if len(parentTasks) == len(self.PrenextWrf.completeTaskSet(parentTasks)): # all its predecessor tasks have been done, just double-check
            processTime =  self.vm_queues[selectedVMind].task_enqueue(self.PrenextTask, self.PrenextTimeStep, self.PrenextWrf)
            self.VMexecHours += processTime/3600                                                                                                              
            self.firstvmWrfLeaveTime[selectedVMind] = self.vm_queues[selectedVMind].get_firstTaskDequeueTime() # return currunt timestap on this machine

            # Update the allocation infos of self.PrenextTask
            if configs.require_estimated_features in [1,2]:
                temp_earliest_finish_times = update_node_weight_and_compute_finish_times(self.PrenextWrf.app, self.PrenextWrf.generateTime,\
                                                                                        self.PrenextTask, processTime, self.PrenextWrf.actual_finish_time)
                for node, estimates in temp_earliest_finish_times.items():
                    self.df.loc[(self.df['Workflow ID'] == self.PrenextWrf.appArivalIndex) & (self.df['Task Index'] == node),\
                                ['Task Finish Time']] = estimates        
            self.df.loc[(self.df['Workflow ID'] == self.PrenextWrf.appArivalIndex) & (self.df['Task Index'] == self.PrenextTask), 
                    ['RemainWd in Wf', 'Task Schedule status', 'Task Ready Time', 'Task Execute Time', 'Task Waiting Time',\
                    'VM ID', 'VM Process Index', 'VM speed', 'VM Utilization']]\
                    = [self.PrenextWrf.remainWorkload, 0, self.PrenextWrf.get_readyTime(self.PrenextTask), processTime,\
                        self.PrenextWrf.get_waitingTime(self.PrenextTask)] + self.PrenextWrf.pendingIndexOnDC[self.PrenextTask]    # + [VM ID', 'VM Process Index', 'VM speed']
            
            self.totalTaskSize += self.PrenextWrf.get_taskSize(self.PrenextTask)

        # 2) Dequeue nextTask
        if self.isDequeue:      # True: the nextTask should be popped out 
            if self.nextisUsr:  # True: the nextTask to be deployed comes from the user queue
                self.nextWrf.update_dequeueTime(self.PrenextTimeStep, self.finishTask)
                _, _ = self.usr_queues[self.PrenextUsr].dequeue() # Here is the actual pop-up of the root task 
                self.firstusrWrfGenTime[self.PrenextUsr] = self.usr_queues[self.PrenextUsr].getFirstWfEnqueueTime() 
                                                            # Updated with the arrival time of the next workflow
                self.usr_sent_pktNum[self.PrenextUsr][self.vm_queues[selectedVMind].get_relativeVMloc()] += 1
                self.stat.add_app_arrival_rate(self.PrenextUsr, self.nextWrf.get_wfType(), self.nextWrf.get_generateTime()) 
            else:               # the nextTask to be deployed comes from the vm queues
                _, _ = self.vm_queues[self.PrenextUsr].task_dequeue() # Here nextTask actually starts to run
                self.firstvmWrfLeaveTime[self.PrenextUsr] = self.vm_queues[self.PrenextUsr].get_firstTaskDequeueTime()
                                                            # Update the current TimeStamp in this machine


        # 3) Update: self.nextTask, and maybe # self.nextWrf, self.finishTask, self.nextUsr, self.nextTimeStep, self.nextisUsr
        temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)   # all successor tasks of the current self.finishTask
                                    # and one successor task has already enqueued

        if len(temp_Children_finishTask) > 0:
            self.dispatchParallelTaskNum += 1

        while True: 

            # self.nextWrf is completed
            while len(temp_Children_finishTask) == 0:  # self.finishTask is the final task of self.nextWrf
                
                if self.nextisUsr:  # for double-check: Default is False
                    # Because it corresponds to self.finishTask, if temp==0, it means it cannot be entry tasks
                    print('self.nextisUsr maybe wrong')
                _, app = self.vm_queues[self.nextUsr].task_dequeue()  
                self.firstvmWrfLeaveTime[self.nextUsr] = self.vm_queues[self.nextUsr].get_firstTaskDequeueTime() 
                        # If there is no task on the VM, math.inf will be returned
                if self.nextWrf.is_completeTaskSet(self.nextWrf.get_allTask()):     # self.nextWrf has been completed
                    respTime = self.nextTimeStep - self.nextWrf.get_generateTime()
                    self.usr_respTime[app.get_originDC()][app.get_wfID()] = respTime/app.totalProcessTime
                    self.usr_received_wrfNum[app.get_originDC()][app.get_wfType()] += 1                    
                    self.completedWF += 1
                    self.remainWrfNum -= 1
                    # ddl_penalty = self.calculate_penalty(app, respTime)
                    # self.SLApenalty += ddl_penalty
                    # self.record_a_completed_workflow(ddl_penalty)
                    # self.all_flowTime.append(respTime)
                    self.all_flowTime[app.appArivalIndex] = respTime

                    complete1Wf = True
                    currentReward -= respTime 
                    # currentReward -= max(self.set.dataset.vmVCPU) * respTime/self.set.dataset.wsetSlowestT[app.appID] 
                    
                    self.finishWf_queues.append([app, self.df[self.df['Workflow ID'] == app.appArivalIndex]])
                    self.df = self.df[self.df['Workflow ID'] != app.appArivalIndex]
                    self.df = self.df.reset_index(drop=True)

                    del app, self.nextWrf                   

                self.get_nextTimeStep()
                if self.nextTimeStep is None:
                    break
                # self.update_VMRemain_infos()
                # self.remove_expired_VMs()                
                self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask)
                self.df.loc[(self.df['Workflow ID'] == self.nextWrf.appArivalIndex) & (self.df['Task Index'] == self.finishTask), 
                            ['Task Schedule status', 'Task Waiting Time', 'Task Finish Time', 'RemainWd in Wf']]\
                            = [0, self.nextWrf.get_waitingTime(self.finishTask), self.nextTimeStep, self.nextWrf.remainWorkload] 
                self.update_ready_tasks(self.nextWrf, self.finishTask)                  
                temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)              

            if self.nextTimeStep is None:
                break

            # Indicates that parallel tasks have not been allocated yet, and len(temp_Children_finishTask)>=1
            if len(temp_Children_finishTask) > self.dispatchParallelTaskNum: 
                to_be_next = None
                while len(temp_Children_finishTask) > self.dispatchParallelTaskNum:
                    temp_nextTask = temp_Children_finishTask[self.dispatchParallelTaskNum]
                    temp_parent_nextTask = self.nextWrf.get_allpreviousTask(temp_nextTask)
                    if len(temp_parent_nextTask) - len(self.nextWrf.completeTaskSet(temp_parent_nextTask)) >0:
                        self.dispatchParallelTaskNum += 1
                    else: 
                        to_be_next = temp_nextTask
                        break

                if to_be_next is not None: 
                    self.nextTask = to_be_next
                    if len(temp_Children_finishTask) - self.dispatchParallelTaskNum > 1:
                        self.isDequeue = False
                    else:
                        self.isDequeue = True
                    break

                else: # Mainly to loop this part
                    _, _ = self.vm_queues[self.nextUsr].task_dequeue() # Actually start running self.nextTask here
                    self.firstvmWrfLeaveTime[self.nextUsr] = self.vm_queues[self.nextUsr].get_firstTaskDequeueTime()
                    self.get_nextTimeStep() 
                    # self.update_VMRemain_infos()
                    # self.remove_expired_VMs()                        
                    self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask) 
                    self.df.loc[(self.df['Workflow ID'] == self.nextWrf.appArivalIndex) & (self.df['Task Index'] == self.finishTask), 
                                ['Task Schedule status', 'Task Waiting Time', 'Task Finish Time', 'RemainWd in Wf']]\
                              = [0, self.nextWrf.get_waitingTime(self.finishTask), self.nextTimeStep, self.nextWrf.remainWorkload]      
                    self.update_ready_tasks(self.nextWrf, self.finishTask)                
                    self.dispatchParallelTaskNum = 0                     
                    if self.nextTimeStep is not None:
                        temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)                                

            else: # i.e., len(temp_Children_finishTask)<=self.dispatchParallelTaskNum
                # self.nextTask is the last imcompleted successor task of the self.finishTask
                if not self.isDequeue:      # Defaults to True
                    print('self.isDequeue maybe wrong')      
                self.get_nextTimeStep()
                # self.update_VMRemain_infos()
                # self.remove_expired_VMs()                    
                self.nextWrf.update_dequeueTime(self.nextTimeStep, self.finishTask)
                self.df.loc[(self.df['Workflow ID'] == self.nextWrf.appArivalIndex) & (self.df['Task Index'] == self.finishTask), 
                        ['Task Schedule status', 'Task Waiting Time', 'Task Finish Time', 'RemainWd in Wf']] \
                        = [0, self.nextWrf.get_waitingTime(self.finishTask), self.nextTimeStep, self.nextWrf.remainWorkload]      
                self.update_ready_tasks(self.nextWrf, self.finishTask)
                self.dispatchParallelTaskNum = 0 # Restart recording the number of successor tasks of self.finishTask
                if self.nextTimeStep is not None:
                    temp_Children_finishTask = self.nextWrf.get_allnextTask(self.finishTask)

        
        self.notNormalized_arr_hist = self.stat.update_arrival_rate_history() ## useless for GP
        if self.nextTimeStep is not None:
            self.update_VMrentInfos()
            self.df.loc[(self.df['Workflow ID'] == self.nextWrf.appArivalIndex) & (self.df['Task Index'] == self.nextTask),\
                        ['Task Schedule status']] = 1 # to_be_allocated 
        self.Makespan =  max(self.vmLatestTime.values())
        
        done = False
        if self.remainWrfNum == 0:
            if len(self.firstvmWrfLeaveTime) == 0:
                done = True
            elif self.firstvmWrfLeaveTime[0] == math.inf and self.firstvmWrfLeaveTime.count(self.firstvmWrfLeaveTime[0]) == len(self.firstvmWrfLeaveTime):
                done = True

        reward_M = self.PreMakespan - self.Makespan
        self.cumulative_reward_M += reward_M
        self.PreMakespan = self.Makespan
        self.PrenextWrf , self.PrenextTask = self.nextWrf, self.nextTask

        if done:
            # reward_M = -self.VMcost-self.SLApenalty
            self.returns_M = self.Makespan   
            self.episode_info = {'makespanOfWf':self.usr_respTime, 'processRate':self.Makespan/self.totalTaskSize, 'makespan':self.returns_M,\
                                  'return':self.cumulative_reward_M, 'Infos':self.finishWf_queues, 'maxVMs': self.maxVms,\
                                    'responseTime': np.mean(self.all_flowTime)}
            # print(self.maxVms)

        if complete1Wf:
            reward = currentReward
        else: 
            reward = 0
        self.maxVms = len(self.df) #max(self.maxVms, len(self.df))

        return reward, done


    # calculate the total VM cost during an episode
    def update_VMcost(self, dc, cpu, add=True):
        if add:
            temp = 1
        else:
            temp = 0
        self.VMcost += temp * self.set.dataset.vmPrice[cpu]      # (self.set.dataset.datacenter[dc][-1])/2 * cpu
        self.VMrentHours += temp


    def calculate_penalty(self, app, respTime):
        appID = app.get_wfType()
        threshold = app.get_Deadline() - app.get_generateTime()
        if respTime < threshold or round(respTime - threshold,5) == 0:
            return 0
        else:
            self.missDeadlineNum += 1
            return 1+self.set.dataset.wsetBeta[appID]*(respTime-threshold)/3600


    def add_self_loops(self, dag):

        selfEdges = []
        for node in dag.nodes():
            selfEdges.append([node, node])

        return np.array(selfEdges).T

    def state_info_construct1(self):  #
        ## All VM Utilization values ​​with empty vm id are replaced with the real-time mean_utl value.
        if configs.require_estimated_features == 1:
            self.df.loc[self.df['VM ID'].isnull(), 'VM Utilization'] = np.mean(self.vm_queues_utl) 
        # Workflow infos: wf_features, wf_edges, wf_masks, candidate_taskID 
        selected_features = ['Task Schedule status','Task Size','Wd2End', 'Task Execute Time',\
                             'Task Finish Time',  'VM speed', 'VM Utilization'] #
        wf_features = self.df.loc[:, selected_features].values 
        wf_edges = []
        wf_masks = []                  
        totalTasks = 0
        for wfID in self.df['Workflow ID'].unique():
            if configs.remove_completed==0:
                array1 = np.array(self.activeWf_queues[wfID].tempDAG.edges()).T
                if array1.size == 0:
                    temp_edges = self.add_self_loops(self.activeWf_queues[wfID].tempDAG) + totalTasks
                else:
                    temp_edges = np.concatenate((array1,  self.add_self_loops(self.activeWf_queues[wfID].tempDAG)), axis=1) + totalTasks
            else:
                array1 = np.array(self.activeWf_queues[wfID].app.edges()).T
                temp_edges = np.concatenate((array1,  self.add_self_loops(self.activeWf_queues[wfID].app)), axis=1) + totalTasks            
            wf_edges.append(temp_edges)
            if configs.remove_completed==0:
                wf_masks.append(self.activeWf_queues[wfID].mask)
            else:
                wf_masks.append(np.full(shape=self.activeWf_queues[wfID].app.number_of_nodes(), fill_value=1, dtype=bool))
            if wfID == self.nextWrf.appArivalIndex:
                candidate_taskID = totalTasks + self.nextTask
            totalTasks += self.activeWf_queues[wfID].taskNum
        wf_edges = np.concatenate(wf_edges, axis=1)
        wf_masks = np.concatenate(wf_masks, dtype=bool) 
        wf_edges = np.flip(wf_edges, axis=0)    
        wf_features[wf_features == None] = 0 

        vm_features = []
        vm_edges = []
        # vm_masks = []
        # totalNodes = 0 
        vm_lastIdx = []  
        for i,vm in enumerate(self.vm_queues):
            temp_vm_edges = vm.get_VM_virtual(self.nextTimeStep, False) 
            if len(temp_vm_edges) > 0:
                temp_idx = self.df.index[(self.df['Workflow ID'] == temp_vm_edges[0][0]) & (self.df['Task Index'] == temp_vm_edges[0][1])].tolist()
                vm_lastIdx.append(temp_idx[0])
                for k in range(1,len(temp_vm_edges)):  
                    temp_idx = self.df.index[(self.df['Workflow ID'] == temp_vm_edges[k][0]) & (self.df['Task Index'] == temp_vm_edges[k][1])].tolist()
                    wf_edges = np.column_stack([wf_edges, [vm_lastIdx[-1], temp_idx[0]]])
                    vm_lastIdx[-1] = temp_idx[0]
            else:
                vm_lastIdx.append([])
            # 准备要添加进wf_features的相应infos
            vm_features.append([vm.get_taskExecuteTime(self.nextTask, self.nextWrf),
                            #    vm.vmQueueTime(),
                               vm.get_taskFinishTime(self.nextTask, self.nextWrf), 
                               vm.cpu,
                               vm.get_vmUtilization(self.nextTimeStep)])
            vm_edges.append([vm_lastIdx[-1], candidate_taskID] if isinstance(vm_lastIdx[-1],int) else [])

        wf_batchs = np.full((len(wf_features),), 0, dtype=np.int64)  

        return (np.array([candidate_taskID]), wf_features.astype(np.float32), wf_edges, wf_masks, wf_batchs,\
                 np.array(vm_features), np.array(vm_edges, dtype=object), np.array(vm_lastIdx,dtype=object), np.array([]))


    def state_info_construct2(self):  # configs.require_estimated_features == 2
        ## 所有vm id空的VM Utilization替换成实时的mean_utl值
        self.df.loc[self.df['VM ID'].isnull(), 'VM Utilization'] = np.mean(self.vm_queues_utl) 
        common_wf_edges = []
        wf_masks = []                  
        totalTasks = 0
        for wfID in self.df['Workflow ID'].unique():
            if configs.remove_completed==0:
                array1 = np.array(self.activeWf_queues[wfID].tempDAG.edges()).T
                if array1.size == 0:
                    temp_edges = self.add_self_loops(self.activeWf_queues[wfID].tempDAG) + totalTasks
                else:
                    temp_edges = np.concatenate((array1,  self.add_self_loops(self.activeWf_queues[wfID].tempDAG)), axis=1) + totalTasks
            else:
                array1 = np.array(self.activeWf_queues[wfID].app.edges()).T
                temp_edges = np.concatenate((array1,  self.add_self_loops(self.activeWf_queues[wfID].app)), axis=1) + totalTasks               

            common_wf_edges.append(temp_edges)
            if configs.remove_completed==0:
                wf_masks.append(self.activeWf_queues[wfID].mask)
            else:
                wf_masks.append(np.full(shape=self.activeWf_queues[wfID].app.number_of_nodes(), fill_value=1, dtype=bool))
            if wfID == self.nextWrf.appArivalIndex:
                candidate_taskID = totalTasks + self.nextTask
            totalTasks += self.activeWf_queues[wfID].taskNum
        common_wf_edges = np.concatenate(common_wf_edges, axis=1)
        wf_masks = np.concatenate(wf_masks, dtype=bool) 
        common_wf_edges = np.flip(common_wf_edges, axis=0)    

        vm_edges = []
        vm_lastIdx = []  # The index of the last node on each vm in wf_features
        for vm in self.vm_queues:
            temp_vm_edges = vm.get_VM_virtual(self.nextTimeStep, False) # Excluding virtual points
            if len(temp_vm_edges) > 0:
                temp_idx = self.df.index[(self.df['Workflow ID'] == temp_vm_edges[0][0]) & (self.df['Task Index'] == temp_vm_edges[0][1])].tolist()
                vm_lastIdx.append(temp_idx[0])
                for k in range(1,len(temp_vm_edges)):  # Processing subsequent pending task edges
                    temp_idx = self.df.index[(self.df['Workflow ID'] == temp_vm_edges[k][0]) & (self.df['Task Index'] == temp_vm_edges[k][1])].tolist()
                    common_wf_edges = np.column_stack([common_wf_edges, [vm_lastIdx[-1], temp_idx[0]]])
                    vm_lastIdx[-1] = temp_idx[0]
            else:
                vm_lastIdx.append([])

            vm_edges.append([vm_lastIdx[-1], candidate_taskID] if isinstance(vm_lastIdx[-1],int) else [])
            

        selected_features = ['Task Schedule status','Task Size','Wd2End', 'Task Execute Time',\
                             'Task Finish Time',  'VM speed', 'VM Utilization']

        wf_features = []
        wf_edges = []
        candidate_indxs = []
        vm_batchs = []
        totalTasks = 0
        for i,vm in enumerate(self.vm_queues): 
            candidate_indxs.append(candidate_taskID + totalTasks)
            temp_df = copy.deepcopy(self.df)   
            temp_df.loc[(temp_df['Workflow ID'] == self.PrenextWrf.appArivalIndex) & (temp_df['Task Index'] == self.PrenextTask),\
                ['Task Execute Time', 'VM speed', 'VM Utilization'] ] = [vm.get_taskExecuteTime(self.PrenextTask, self.PrenextWrf),
                                                                         vm.cpu,
                                                                         vm.get_vmUtilization(self.nextTimeStep)]
            temp_dag = copy.deepcopy(self.PrenextWrf.app)
            temp_actual_finish_time = copy.deepcopy(self.PrenextWrf.actual_finish_time)
            temp_actual_finish_time[self.PrenextTask] = vm.get_taskFinishTime(self.PrenextTask, self.PrenextWrf)
            temp_earliest_finish_times = update_node_weight_and_compute_finish_times(temp_dag,
                                                                                     self.PrenextWrf.generateTime,
                                                                                     self.PrenextTask,
                                                                                     vm.get_taskExecuteTime(self.PrenextTask, self.PrenextWrf),
                                                                                     temp_actual_finish_time)
            for node, estimates in temp_earliest_finish_times.items():
                temp_df.loc[(temp_df['Workflow ID'] == self.PrenextWrf.appArivalIndex) & (temp_df['Task Index'] == node),\
                            ['Task Finish Time']] = estimates  
            temp_wf_features = temp_df.loc[:, selected_features].values         
            temp_wf_features[temp_wf_features == None] = 0 
            wf_features.append(temp_wf_features)
            vm_batchs.append( np.full((len(temp_wf_features),), i, dtype=np.int32) )

            if len(vm_edges[i])>0:
                wf_edges.append( np.column_stack([common_wf_edges, vm_edges[i]]) + totalTasks)
            else:
                wf_edges.append( common_wf_edges + totalTasks )

            totalTasks += wf_features[-1].shape[0]

        candidate_indxs = np.array(candidate_indxs)    
        wf_features = np.concatenate(wf_features, axis=0).astype(np.float32)   
        wf_edges = np.concatenate(wf_edges, axis=1)  
        wf_masks = np.tile(wf_masks, len(candidate_indxs)) 
        wf_batchs = np.full((len(wf_features),), 0, dtype=np.int32)  
        vm_batchs = np.concatenate(vm_batchs, axis=0)
        
        return (candidate_indxs, wf_features, wf_edges, wf_masks, wf_batchs,\
                 np.array([]), np.array(vm_edges, dtype=object), np.array(vm_lastIdx,dtype=object),vm_batchs)


    def state_info_construct_include_virtualNode(self):  # without estimated features

        # Workflow infos: wf_features, wf_edges, wf_masks, candidate_taskID
        selected_features = ['Task Schedule status','Task Size','Wd2End', 'Task Execute Time',\
                              'Task Waiting Time', 'Task Finish Time',  'VM speed', 'VM Utilization']
        wf_features = self.df.loc[:, selected_features].values #astype(np.float32)
        wf_edges = []
        wf_masks = []                  
        totalTasks = 0
        for wfID in self.df['Workflow ID'].unique():
            array1 = np.array(self.activeWf_queues[wfID].tempDAG.edges()).T
            if array1.size == 0:
                temp_edges = self.add_self_loops(self.activeWf_queues[wfID].tempDAG) + totalTasks
            else:
                temp_edges = np.concatenate((array1,  self.add_self_loops(self.activeWf_queues[wfID].tempDAG)), axis=1) + totalTasks
            wf_edges.append(temp_edges)
            wf_masks.append(self.activeWf_queues[wfID].mask)
            if wfID == self.nextWrf.appArivalIndex:
                candidate_taskID = totalTasks + self.nextTask
            totalTasks += self.activeWf_queues[wfID].taskNum
        wf_edges = np.concatenate(wf_edges, axis=1)
        wf_masks = np.concatenate(wf_masks, dtype=bool) ## indicate uncompleted tasks
        wf_edges = np.flip(wf_edges, axis=0)    # The order of becoming a predecessor
        wf_features[wf_features == None] = 0 

        vm_features = []
        vm_edges = []
        # vm_masks = []
        # totalNodes = 0 
        vm_lastIdx = []  # The index of the last node on each vm in wf_features
        for i,vm in enumerate(self.vm_queues):
            temp_vm_edges = vm.get_VM_virtual(self.nextTimeStep)
            wf_features = np.vstack([wf_features, temp_vm_edges[0]])
            wf_masks = np.append(wf_masks, True)
            vm_lastIdx.append(len(wf_features)-1)
            wf_edges = np.column_stack([wf_edges, [vm_lastIdx[-1], vm_lastIdx[-1]]])
            for k in range(1,len(temp_vm_edges)):  # Processing subsequent pending task edges
                temp_idx = self.df.index[(self.df['Workflow ID'] == temp_vm_edges[k][0]) & (self.df['Task Index'] == temp_vm_edges[k][1])].tolist()
                wf_edges = np.column_stack([wf_edges, [vm_lastIdx[-1], temp_idx[0]]])
                vm_lastIdx[-1] = temp_idx[0]
            # Prepare the corresponding infos to be added to wf_features
            vm_features.append([vm.get_taskExecuteTime(self.nextTask, self.nextWrf),
                               vm.vmQueueTime(),
                               vm.get_taskFinishTime(self.nextTask, self.nextWrf), 
                               vm.cpu,
                               vm.get_vmUtilization(self.nextTimeStep)])
            vm_edges.append([vm_lastIdx[-1], candidate_taskID])

        wf_batchs = np.full((len(wf_features),), 0, dtype=np.int64)  # Indicates which graph these tasks belong to

        return (np.array([candidate_taskID]), wf_features.astype(np.float32), wf_edges, wf_masks, wf_batchs,\
                 np.array(vm_features), np.array(vm_edges), np.array(vm_lastIdx), np.array([]))
    

    def gp_feature_construct(self):

        '''
        features:
        0.	TS: task_size of a task
        1.	RW: remain workload in the workflow
        2.	ET: execute_time_period of a task on this VM
        3.	FT: finish_time = start_time + ET
        4.  CU: compute unit of a VM
        5.	UL: utilization of a VM 

        '''
        obs = np.zeros((self.vmNum, 6))
        features = self.df.loc[(self.df['Workflow ID'] == self.nextWrf.appArivalIndex) &\
                                (self.df['Task Index'] == self.nextTask), ['Task Size','Wd2End']] 
        obs[:,:2] = features.values

        for i,vm in enumerate(self.vm_queues):
            obs[i,2] = vm.get_taskExecuteTime(self.nextTask, self.nextWrf)
            obs[i,3] = vm.vmLatestTime() + obs[i,2]   #get_taskFinishTime(self.nextTask, self.nextWrf)
            obs[i,4] = vm.cpu
            obs[i,5] = vm.get_vmUtilization(self.nextTimeStep)
 
        return obs

    def HEFT(self):

        priorities = []
        for vm in self.vm_queues:
            priorities.append(vm.vmLatestTime() + vm.get_taskExecuteTime(self.nextTask, self.nextWrf) ) 
        min_indices = np.where(priorities == np.min(priorities))[0]

        return np.random.choice(min_indices)

    def EST(self):

        priorities = []
        for vm in self.vm_queues:
            priorities.append(vm.vmLatestTime()) 

        min_indices = np.where(priorities == np.min(priorities))[0]

        return np.random.choice(min_indices)

    def PEFT(self):
        shortPath2end = self.nextWrf.get_path2end(self.nextTask)
        priorities = []
        for vm in self.vm_queues:
            priorities.append(vm.vmLatestTime() + vm.get_taskExecuteTime(self.nextTask, self.nextWrf) + shortPath2end/vm.cpu  ) 

        min_indices = np.where(priorities == np.min(priorities))[0]

        return np.random.choice(min_indices)        
    

    def esrl_feature_construct(self):

        '''
        features:
        0.	Number of child tasks: childNum
        1.	Completion ratio: completionRatio
        2,3.Workflow arrival rate: arrivalRate (a vector of historical arrivalRate)
        4.	ET: execute_time_period of a task on this VM
        5.	FT: finish_time = start_time + ET
        6.  CU: compute unit of a VM
        7.	UL: utilization of a VM 

        '''
        obs = np.zeros((self.vmNum, 8))
        obs[:,0] = len(self.nextWrf.get_allnextTask(self.nextTask)) 
        obs[:,1] = self.nextWrf.get_completeTaskNum() / self.nextWrf.get_totNumofTask()
        obs[:,2] = np.sum(np.sum(self.notNormalized_arr_hist, axis=0), axis=0)[0]
        obs[:,3] = np.sum(np.sum(self.notNormalized_arr_hist, axis=0), axis=0)[1]

        for i,vm in enumerate(self.vm_queues):
            obs[i,4] = vm.get_taskExecuteTime(self.nextTask, self.nextWrf)
            obs[i,5] = vm.vmLatestTime() + obs[i,2]   #get_taskFinishTime(self.nextTask, self.nextWrf)
            obs[i,6] = vm.cpu
            obs[i,7] = vm.get_vmUtilization(self.nextTimeStep)
 
        return obs