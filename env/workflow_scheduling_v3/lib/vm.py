import numpy as np
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from env.workflow_scheduling_v3.lib.simqueue import SimQueue
# from workflow_scheduling.env.poissonSampling import one_sample_poisson
import math
# import heapq


class VM:
    def __init__(self, id, cpu, dcind, abind, t, rule): 
        ##self, vmID, vmCPU, dcID, dataset.datacenter[dcid][0], self.nextTimeStep, task_selection_rule
        self.vmid = id
        self.cpu = cpu
        self.loc = dcind  # the index of dc (ranging from 0 to dcNum)
        self.abloc = abind  # the relative index of dc in the topology (ranging from 0 to usrNum)
        self.vmQueue = SimQueue()  # store the apps waiting to be processed
        self.currentTimeStep = t  # record the leave time of the first processing app; 是处理完self.processingtask后的时间
        self.rentStartTime = t
        self.rentEndTime = t        # 当前最晚的结束时间
        self.processingApp = None  # store the app with the highest priority
        self.processingtask = None  # the task associated with the highest priority app
        self.totalProcessTime = 0  # record the total processing time required for all queuing tasks
        self.pendingTaskTime = 0    # 是去除self.processingtask后的总pending时间
        # self.pendingTaskNum = 0
        self.taskSelectRule = rule
        self.currentQlen = 0
        self.vm_utilization = 0
        self.processingIndex = 0
        self.vmInfos = [] # []
        self.wfInfos = []

    def get_utilization(self, app, task):
        numOfTask = self.totalProcessTime / self.get_taskExecuteTime(task, app)
        util = numOfTask/self.get_capacity(app, task) 
        return util  ## == self.totalProcessTime / 60*60

    def get_capacity(self, app, task):
        return 60*60 / self.get_taskExecuteTime(task, app)  # how many tasks can processed in one hour

    def get_vmid(self):
        return self.vmid

    def get_cpu(self):
        return self.cpu

    def get_relativeVMloc(self):
        return self.loc

    def get_absoluteVMloc(self):
        return self.abloc

    ## self-defined
    def cal_priority(self, task, app):
        enqueueTime = app.get_enqueueTime(task)
        return enqueueTime        

    def get_firstTaskEnqueueTimeinVM(self):
        if self.processingApp is None:
            return math.inf
        return self.processingApp.get_enqueueTime(self.processingtask)

    def get_firstTaskDequeueTime(self):
        if self.get_pendingTaskNum() > 0:
            return self.currentTimeStep
        else:
            return math.inf

    def get_firstDequeueTask(self):
        return self.processingApp, self.processingtask

    # how long a new task needs to wait if it is assigned
    def get_pendingTaskNum(self):
        if self.processingApp is None:
            return 0
        else:
            return self.vmQueue.qlen()+1  # 1 is needed

    def get_taskExecuteTime(self, task, app):
        return app.get_taskProcessTime(task)/self.cpu

    def get_taskWaitTime_beforeExecuted(self, task, app):
        # return self.currentTimeStep + self.pendingTaskTime - app.get_readyTime(task) 
        gap2ready = max(self.currentTimeStep, app.get_readyTime(task)) - app.get_readyTime(task)
        return gap2ready + self.pendingTaskTime
    
    def get_taskWaitTime_afterExecuted(self, task, app):        
        return app.get_enqueueTime(task) - app.get_readyTime(task) # startTime - readyTime

    def get_taskFinishTime(self, task, app):  ## 不对，有可能会self.currentTimeStep有数，self.processingtask = None 而又新进来一个task
        if task in app.enqueueTime:
            currentTime = max(self.currentTimeStep, app.get_enqueueTime(task))
        else: 
            currentTime = self.currentTimeStep
        executeTime = self.get_taskExecuteTime(task, app)
        return  currentTime + self.pendingTaskTime + executeTime

    def get_vmUtilization(self, currentTime):
        occupyTime =  max(self.currentTimeStep + self.pendingTaskTime, currentTime) - self.rentStartTime
        return self.totalProcessTime/occupyTime if abs(occupyTime) > 1e-8 else 0

    def get_VM_virtual(self, currentTime, virtual=True):
        features = []
        if virtual:
            features.append([0, 0, 0, 0, 0, 0, self.cpu, self.get_vmUtilization(currentTime)])
        if self.processingtask is not None:
            features.append([ self.processingApp.appArivalIndex , self.processingtask ])
            if self.wfInfos != []: 
                features.extend(self.wfInfos)
        return features

    def get_VMinfos(self, currentTime):  
        features = []
        # Add: task execution time, task waiting time, task finish time
        if self.processingtask is None:
            features.append([0, 0, max(self.currentTimeStep, currentTime)])
        else: 
            # self.processingtask
            features.append([self.get_taskExecuteTime(self.processingtask, self.processingApp),\
                             self.get_taskWaitTime_afterExecuted(self.processingtask, self.processingApp), self.currentTimeStep])
            # other tasks in self.vmQueue.queue
            if self.vmInfos != []: 
                features.extend(self.vmInfos) 
        # Add: VM speed, VM utilization
        speed = np.full((len(features),1), self.cpu)
        utilization = np.full((len(features),1), self.get_vmUtilization(currentTime))
        # Concat
        outputs = np.hstack((np.array(features), speed, utilization))
        return outputs

    def task_enqueue(self, task, enqueueTime, app):
        executeTime = self.get_taskExecuteTime(task, app)   # execution time
        self.totalProcessTime += executeTime                
        self.currentQlen += 1 #self.get_pendingTaskNum()

        app.update_executeTime(executeTime, task)
        app.update_enqueueTime(enqueueTime, task, self.vmid)
        self.vmQueue.enqueue(app, enqueueTime, task, self.vmid, self.processingIndex) # last is priority
        self.processingIndex +=1

        waitTime = self.get_taskWaitTime_beforeExecuted(task, app)
        app.update_waitingTime(waitTime, task)
        self.vmInfos.append([executeTime, waitTime, self.get_taskFinishTime(task, app) ])
        self.wfInfos.append([ app.appArivalIndex, task ])
        app.actual_finish_time[task] = self.get_taskFinishTime(task, app) 
        self.pendingTaskTime += executeTime
        if self.processingApp is None:
            self.process_task()

        app.update_pendingIndexVM(task, self.vmid, self.processingIndex, self.cpu, self.get_vmUtilization(enqueueTime))
        return executeTime

    def task_dequeue(self):
        task, app = self.processingtask, self.processingApp 
        qlen = self.vmQueue.qlen()
        if qlen == 0:   # 下一个task为空
            self.processingApp = None
            self.processingtask = None
        else:
            self.process_task()     # update next self.processingtask, self.processingApp

        return task, app 

    def process_task(self): #
        self.processingtask, self.processingApp = self.vmQueue.dequeue() 
            # Pop and return the smallest item from the heap, the popped item is deleted from the heap
        enqueueTime = self.processingApp.get_enqueueTime(self.processingtask)
        executeTime = self.processingApp.get_executeTime(self.processingtask)

        taskStratTime = max(enqueueTime , self.currentTimeStep)
        finishTime = taskStratTime +executeTime
        self.processingApp.actual_finish_time[self.processingtask] = finishTime 

        self.processingApp.update_enqueueTime(taskStratTime, self.processingtask, self.vmid)
        self.pendingTaskTime -= executeTime
        # self.processingApp.update_pendingIndexVM(self.processingtask, self.vmid, self.processingIndex)
        # self.processingIndex +=1
        self.currentTimeStep = finishTime
        self.currentQlen-=1
        del self.vmInfos[0]
        del self.wfInfos[0]

    def vmQueueTime(self): 
        return max(round(self.pendingTaskTime,3), 0)

    def vmTotalTime(self): 
        return self.totalProcessTime
    
    def vmLatestTime(self): 
        # return self.totalProcessTime+self.rentStartTime    
        return self.currentTimeStep + self.pendingTaskTime
    
    def get_vmRentEndTime(self):
        return self.rentEndTime
    
    def update_vmRentEndTime(self, time):
        self.rentEndTime += time

    ## real_waitingTime in dual-tree = currentTime - enqueueTime
    def get_real_taskWaitingTime(self, app, task): 
        waitingTime = self.currentTimeStep - app.get_enqueueTime(task)
        return waitingTime
