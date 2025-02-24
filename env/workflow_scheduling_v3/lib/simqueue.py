import logging
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
import heapq
import math

class SimQueue:
    def __init__(self):
        self.queue = []

    def qlen(self):
        return len(self.queue)

    def enqueue(self, wkf, t, task, vmID, priority): 
        ## self, workflow, enqueueTime, task, vmID, priority
        if task is None:
            wkf.update_enqueueTime(t, task, vmID)
        heapq.heappush(self.queue, (priority, task, wkf)) # Add new task information in self.queue

    def dequeue(self):
        if len(self.queue) > 0:
            _, task, wkf = heapq.heappop(self.queue) 
                # Pop and return the smallest item from the heap, the popped item is deleted from the heap 
            return task, wkf
        else:
            logging.error("queue is empty")
            sys.exit(1)

    def getFirstWfEnqueueTime(self):
        if len(self.queue) > 0:
            _, task, firstPkt = self.queue[0]
            enqueueTime = firstPkt.get_readyTime(task)
            return enqueueTime
        else:
            return math.inf

    def getFirstWf(self):
        if len(self.queue) > 0:
            _, task, firstPkt = self.queue[0]   ## based on generateTime

            return firstPkt, task
        else:
            return None, None

