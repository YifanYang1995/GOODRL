from collections import deque
import numpy as np

class Stats:
    def __init__(self, setting, arr_rate_map_data_struct="Array"):
        self.history_len = setting.history_len
        self.arrival_rate_update_interval = setting.arrival_rate_update_interval

        self.usr_no = setting.usrNum
        self.dc_no = setting.dcNum
        self.app_no = setting.WorkflowNum

        self.arr_rate_index = deque([])
        self.app_arr_rate_map = np.zeros((self.usr_no, len(setting.dataset.wset), self.history_len))
       
        self._init_arr_rate_map()

    def _init_arr_rate_map(self):
        for his in range(self.history_len):
            self.arr_rate_index.append(his-self.history_len)

    def check_arr_rate_index(self, ind_new_orig):
        if ind_new_orig not in self.arr_rate_index: 
            ind_old = self.arr_rate_index.popleft()
            intervals = ind_new_orig - ind_old
            while True:
                ind_new = ind_old + self.history_len
                self.arr_rate_index.append(ind_new)
                self.app_arr_rate_map = np.roll(self.app_arr_rate_map, -1, axis=-1)  # shift all elements to left by offset=1
                self.app_arr_rate_map[:,:,-1] = 0  # the value under ind_new is 0

                if intervals == self.history_len:
                    break
                else:       # intervals > self.history_len
                    ind_old = self.arr_rate_index.popleft()
                    intervals = ind_new_orig - ind_old      ## continue to move self.app_arr_rate_map to left


    def add_app_arrival_rate(self, usr, app, arrival_time):
        ind_new = int(arrival_time/self.arrival_rate_update_interval)
        if ind_new < self.arr_rate_index[0]:
            return
        self.check_arr_rate_index(ind_new)
        self.app_arr_rate_map[usr][app][self.arr_rate_index.index(ind_new)] += 1
 
    def update_arrival_rate_history(self):
        return self.app_arr_rate_map
