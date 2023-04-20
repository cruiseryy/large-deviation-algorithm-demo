import numpy as np
from matplotlib import pyplot as plt
from time import time

class mc_lagone:

    def __init__(self) -> None:
        pass

    
    def mc_simulation(self, k):
        prcp = np.mean(np.loadtxt('MC_RAIN_DEMO_new/sta_daily.csv'), axis=1)
        mm  = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        data = np.zeros([39, 90])
        cur = np.sum(mm[:11])
        for i in range(39):
            data[i, :] = prcp[cur:cur+90]
            cur += 365

        thres = 1
        data[np.where(data<=thres)] = 0

        data = data[np.argsort(np.sum(data, axis=1)),:]
        K = k
        if K == 1:
            data[-1,:] = data[-2,:]
        data = data[:39, :]
        prcp = data[np.where(data > thres)]

        p00 = p01 = p10 = p11 = 0
        for i in range(39):
            for j in range(2, 90):
                if data[i, j-1] <= thres and data[i, j] <= thres:
                    p00 += 1
                if data[i, j-1] <= thres and data[i, j] > thres:
                    p01 += 1 
                if data[i, j-1] > thres and data[i, j] <= thres:
                    p10 += 1
                if data[i, j-1] > thres and data[i, j] > thres:
                    p11 += 1 
        p00, p01 = p00/(p00+p01), p01/(p00+p01)
        p10, p11 = p10/(p10+p11), p11/(p10+p11)
        p1 = len(prcp) / data.shape[0] / data.shape[1]
        p0 = 1 - p1
        ref = np.sum(prcp) / data.shape[0] / data.shape[1]

        nn = 1000
        wd = np.zeros([nn, 39])
        for k in range(nn):
            t1 = time()
            for y in range(39):
                traj = np.zeros([90, ])
                d = np.random.rand()
                if d < p1:
                    traj[0] = np.random.choice(prcp)
                for j in range(1, 90):
                    d = np.random.rand()
                    if traj[j-1] == 0:
                        if d < p01:
                            traj[j] = np.random.choice(prcp)
                    else:
                        if d < p11: 
                            traj[j] = np.random.choice(prcp)
                wd[k, y] = np.sum(traj)
            wd[k, :] = sorted(wd[k, :], reverse=False)
            t2 = time()
            print((k, t2-t1))

        wd0 = np.zeros([39,])
        for y in range(39):
            wd0[y] = np.sum(data[y, :])

        wd0 = sorted(wd0, reverse=False)

        # fig, ax = plt.subplots()
        # alpha = 1 
        # beta = 1
        # for y in range(39):
        #     low = np.quantile(wd[:, y], 0.05)
        #     upp = np.quantile(wd[:, y], 0.95)
        #     ax.plot([low, upp],[(y+alpha)/(39+beta), (y+alpha)/(39+beta)], color='black')
        #     ax.scatter(wd0[y], (y+alpha)/(39+beta), marker='D', color='red')
        #     ax.scatter(np.mean(wd[:, y]), (y+alpha)/(39+beta), marker='o', color='black')

        return wd, wd0

# pause = 1

if __name__ == '__main__':

    test = mc_lagone()
    wd, wd0 = test.mc_simulation(k=1)

    fig, ax = plt.subplots()
    K = 1
    alpha = 1 
    beta = 1
    for y in range(39):
        low = np.quantile(wd[:, y], 0.05)
        upp = np.quantile(wd[:, y], 0.95)
        ax.plot([low, upp],[(y+alpha)/(39+beta), (y+alpha)/(39+beta)], color='black')
        ax.scatter(wd0[y], (y+alpha)/(39+beta), marker='D', color='red')
        ax.scatter(np.mean(wd[:, y]), (y+alpha)/(39+beta), marker='o', color='black')
    pause = 1