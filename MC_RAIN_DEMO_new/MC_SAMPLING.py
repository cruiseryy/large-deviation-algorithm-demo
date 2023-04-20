import numpy as np
import collections

class mc_sampler:
    def __init__(self, thres=1) -> None:
        prcp = np.mean(np.loadtxt('MC_RAIN_DEMO_new/sta_daily.csv'), axis=1)
        mm  = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        data = np.zeros([39, 90])
        cur = np.sum(mm[:11])
        for i in range(39):
            data[i, :] = prcp[cur:cur+90]
            cur += 365
        self.thres = thres
        data[np.where(data<=self.thres)] = 0
        self.data = data[np.argsort(np.sum(data, axis=1)),:]
        
    
    def modify(self, flag=0):
        if flag == 1:
            self.data[-1,:] = self.data[-2,:]

    def get_para(self):
        pp = [[[0]*2 for _ in range(2)] for _ in range(2)]
        for i in range(39):
            for j in range(3, 90):
                f1 = 0 if self.data[i, j-2] <= self.thres else 1
                f2 = 0 if self.data[i, j-1] <= self.thres else 1
                f3 = 0 if self.data[i, j] <= self.thres else 1
                pp[f1][f2][f3] += 1
        ppp = [[0]*2 for _ in range(2)]
        for i in range(2):
            for j in range(2):
                ppp[i][j] = pp[i][j][0] + pp[i][j][1]
                pp[i][j][0], pp[i][j][1] = pp[i][j][0]/ppp[i][j], pp[i][j][1]/ppp[i][j]
        pause = 1

if __name__ == "__main__":
    test = mc_sampler()
    test.modify(flag=0)
    test.get_para()

        
    