import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import bisect
from MC_SAMPLING import mc_sampler2
from time import time

class GEN: 
    def __init__(self, g1, g2, pd) -> None:
        self.g1 = g1
        self.g2 = g2
        self.pd = pd

    def generate(self, ic=[], dt=5):
        d = np.random.rand()
        if d <= self.pd:
            traj = self.g1.generate(dt=dt, ic=ic)
        else:
            traj = self.g2.generate(dt=dt, ic=ic)
        return traj
    def generate1(self, ic, dt=5):
        traj = self.g1.generate(dt=dt, ic=ic)
        return traj

    def generate2(self, ic, dt=5):
        traj = self.g2.generate(dt=dt, ic=ic)
        return traj

# class LDS_demo:
#     def __init__(self,
#                  k = 0.1, 
#                  dt = 5, 
#                  T = 90, 
#                  N = 1000, 
#                  gen = None,
#                  flag = 1):
        
#         self.k = k
#         self.dt = dt
#         self.T = T 
#         self.N = N
#         self.gen = gen
#         self.flag = flag
#         self.thres = 7.07842
#         self.traj = np.zeros([self.N, self.T])
#         self.R = np.zeros([self.T//self.dt, ])
#         self.res = np.zeros([self.N, 2])

#         return
    
#     def evalute(self):
#         tot = [(self.score(self.traj[j,:]), j) for j in range(self.N)]
#         tot.sort(key = lambda x: x[0], reverse=True)
        
#         dp = 1.0 / self.N
#         if self.k == 0:
#             alter_p = [1]*self.N
#         else:
#             lambda_ = np.sum(np.log(self.R))
#             alter_p = [np.exp(-self.k * tot[j][0])*np.exp(lambda_) for j in range(self.N)]
#         self.res[0, :] = [alter_p[0]*dp, tot[0][0]]
#         for j in range(1, self.N):
#             self.res[j, :] = [self.res[j-1, 0] + dp*alter_p[j], tot[j][0]] 
#         return self.res
    
#     def update(self):
#         if self.k == 0:
#             for j in range(self.N):
#                 self.traj[j, :] = self.gen.generate(dt=self.T, ic=[])
#         else:
#             self.state = [0]*self.N
#             tmp_traj = np.zeros([self.N, self.dt])
#             for i in range(self.T // self.dt):
#                 for j in range(self.N):
#                     if i == 0:
#                         ic = []
#                     else:
#                         ic = [new_traj[j, -1]]
#                     tmp_traj[j, :] = self.gen.generate(dt=self.dt, ic=ic)
#                 new_traj, tmpr = self.restart(tmp_traj)
#                 self.R[i] = tmpr
#                 self.traj[:, i*self.dt: (i+1)*self.dt] = new_traj
#         # tmp_traj = np.zeros([self.N, self.dt])
#         # for i in range(self.T // self.dt):
#         #     for j in range(self.N):
#         #         if i == 0:
#         #             ic = []
#         #         else:
#         #             ic = [new_traj[j, -1]]
#         #         tmp_traj[j, :] = self.gen.generate(dt=self.dt, ic=ic)
#         #     new_traj, tmpr = self.restart(tmp_traj)
#         #     self.R[i] = tmpr
#         #     self.traj[:, i*self.dt: (i+1)*self.dt] = new_traj
#         return 

#     def restart(self, traj):
#         n, m = traj.shape
#         weights = np.zeros([n, ])
#         new_traj = np.zeros([n, m])
        
#         for j in range(n):
#             weights[j] = np.exp(self.k * self.score(traj[j,:]))
#         R = np.mean(weights)
#         weights /= R

#         tmpcdf = np.zeros([n,])
#         tmpcdf[0] = weights[0]
#         for j in range(1, n):
#             tmpcdf[j] = tmpcdf[j-1] + weights[j]
#         tmpcdf /= tmpcdf[-1]

#         for j in range(n):
#             new_traj[j, :] = traj[bisect.bisect(tmpcdf, np.random.rand()), :]
#         return new_traj, R
    
#     def score(self, traj): 
#         # daily rainfall = 7.078424908424909
#         n = len(traj)
#         return self.thres*n - np.sum(traj)
#         # return np.sum(traj)

class LDS_demo:
    def __init__(self,
                 k = 0.1, 
                 dt = 5, 
                 T = 90, 
                 N = 1000, 
                 gen = None,
                 flag = 1):
        
        self.k = k
        self.dt = dt
        self.T = T 
        self.N = N
        self.gen = gen
        self.flag = flag
        self.thres = 7.07842
        self.traj = np.zeros([self.N, self.T])
        self.R = np.zeros([self.T//self.dt, 2])
        self.res = np.zeros([self.N, 2])

        return
    
    def evalute(self):
        tot = [(self.score(self.traj[j,:]), j) for j in range(self.N)]
        tot.sort(key = lambda x: x[0], reverse=True)
        
        dp = 1.0 / self.N
        alter_p = [1]*self.N
        if self.k == 0:
            alter_p = [1]*self.N
        else:
            lambda_d, lambda_w = np.sum(np.log(self.R[:,0])), np.sum(np.log(self.R[:,1]))
            for j in range(self.N):
                if tot[j][1] < self.nd:
                    alter_p[j] = np.exp(-self.k * tot[j][0])*np.exp(lambda_d)
                else:
                    alter_p[j] = np.exp(-self.k * tot[j][0])*np.exp(lambda_w)
        self.res[0, :] = [alter_p[0]*dp, tot[0][0]]
        for j in range(1, self.N):
            self.res[j, :] = [self.res[j-1, 0] + dp*alter_p[j], tot[j][0]] 
        return self.res
    
    def update(self):
        # if self.k == 0:
        #     for j in range(self.N):
        #         self.traj[j, :] = self.gen.generate(dt=self.T, ic=[])
        # else:
            
        nd = int(self.N*self.gen.pd)
        nw = self.N - nd
        self.nd = nd
        self.nw = nw
        traj_d = np.zeros([nd, self.dt])
        traj_w = np.zeros([nw, self.dt])
        for i in range(self.T // self.dt):
            for j in range(nd):
                if i == 0:
                    ic = []
                else:
                    ic = [new_traj_d[j, -1]]
                traj_d[j, :] = self.gen.generate1(dt=self.dt, ic=ic)
            new_traj_d, tmpr_d = self.restart(traj_d)

            for j in range(nw):
                if i == 0:
                    ic = []
                else:
                    ic = [new_traj_w[j, -1]]
                traj_w[j, :] = self.gen.generate2(dt=self.dt, ic=ic)
            new_traj_w, tmpr_w = self.restart(traj_w)
            self.R[i, :] = tmpr_d, tmpr_w
            self.traj[:nd, i*self.dt: (i+1)*self.dt] = new_traj_d
            self.traj[nd:, i*self.dt: (i+1)*self.dt] = new_traj_w
        return 

    def restart(self, traj):
        n, m = traj.shape
        weights = np.zeros([n, ])
        new_traj = np.zeros([n, m])
        
        for j in range(n):
            weights[j] = np.exp(self.k * self.score(traj[j,:]))
        R = np.mean(weights)
        weights /= R

        tmpcdf = np.zeros([n,])
        tmpcdf[0] = weights[0]
        for j in range(1, n):
            tmpcdf[j] = tmpcdf[j-1] + weights[j]
        tmpcdf /= tmpcdf[-1]

        for j in range(n):
            new_traj[j, :] = traj[bisect.bisect(tmpcdf, np.random.rand()), :]
        return new_traj, R
    
    def score(self, traj): 
        # daily rainfall = 7.078424908424909
        n = len(traj)
        return self.thres*n - np.sum(traj)
        # return np.sum(traj)
    
if __name__ == '__main__':
    np.random.seed(777)

    # to prepare the two separate training sets for the MC samplers
    prcp = np.mean(np.loadtxt('large-deviation-algorithm-demo/MC_RAIN_DEMO_new/sta_daily.csv'), axis=1)
    mm  = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    data = np.zeros([39, 90])
    cur = np.sum(mm[:11])
    for i in range(39):
        data[i, :] = prcp[cur:cur+90]
        cur += 365

    thres = 1
    data[np.where(data<=thres)] = 0

    data = data[np.argsort(np.sum(data, axis=1)),:]
    tot = np.sum(data, axis=1)

    model = GaussianMixture(2).fit(np.log(tot).reshape(-1, 1))
    w1, w2 = model.weights_
    sigma1, sigma2  = np.sqrt(model.covariances_)
    mu1, mu2 = model.means_

    # fig0, ax0 = plt.subplots()
    # ax0.hist(np.log(tot))
    # ax0.set_xlabel('log total prcp (mm)')
    # ax0.set_ylabel('frequency')
    # fig0.savefig('MC_RAIN_DEMO_new/histogram.pdf')
    # plt.close(fig0)
    # pause = 1

    # find the intersection in a brute force way 
    txx = np.linspace(6, 6.4, 10000)
    ppx1 = 1/np.sqrt(2*np.pi) /sigma1 * np.exp(-(txx - mu1)**2/2/sigma1**2) * w1
    ppx2 = 1/np.sqrt(2*np.pi) /sigma2 * np.exp(-(txx - mu2)**2/2/sigma2**2) * w2
    diff = np.abs(ppx1 - ppx2).reshape(-1)
    dvd0 = txx[np.where(diff == np.min(diff))]

    dvd = np.exp(dvd0)
    dry = np.sum(tot <= dvd)
    wet = np.sum(tot > dvd)

    pd, pw = min(w1, w2), max(w1, w2)
    idx = bisect.bisect(tot, dvd)
    datad = data[:idx, :]
    dataw = data[idx:, ]

    # mc1 = the drier sampler, mc2 = the wetter sampler
    # mc_sampler = lag-2 MC 
    # mc_sampler = lag-1 MC
    # fit the two MC generators 
    mc1 = mc_sampler2(thres=1, data=datad)
    mc1.get_para()
    mc2 = mc_sampler2(thres=1, data=dataw)
    mc2.get_para()
    # to make things less messy, i used one GEN to call both MC generators
    gg = GEN(mc1, mc2, pd)

    control_run = LDS_demo(gen=gg, k=0, N=10000)
    control_run.update()
    control_run.evalute()
    alter_run = LDS_demo(gen=gg, k=0.005, N=100)
    alter_run.update()
    alter_run.evalute()

    freq = [(i+1)/39 for i in range(39)]
    freq = np.array(freq)
    tot.sort()
    tot = 7.07842*90 - tot
    fig, ax = plt.subplots()
    ax.plot(1/control_run.res[:,0], 7.07842*90 - control_run.res[:,1], color ='k', label='mc_k=0')
    ax.scatter(1/alter_run.res[:,0], 7.07842*90 - alter_run.res[:,1], label='mc_k=0.005')

    ax.scatter(1/freq, 7.07842*90 - tot,marker='D', label='obs')
    ax.set_xscale('log')
    ax.set_xlabel('return period (yr)')
    ax.set_ylabel('total rainfall (mm)')
    ax.legend()
    fig.savefig('large-deviation-algorithm-demo/MC_RAIN_DEMO_new/validation.pdf')
    
    pause = 1






