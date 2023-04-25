import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import bisect
from MC_SAMPLING import mc_sampler2
from time import time

class data_loader:

    def __init__(self, data_path='demo_new/sta_daily.csv', thres=1) -> None:
        
        prcp = np.mean(np.loadtxt(data_path), axis=1)
        mm  = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        data = np.zeros([39, 90])
        cur = np.sum(mm[:11])
        for i in range(39):
            data[i, :] = prcp[cur:cur+90]
            cur += 365

        data[np.where(data<=thres)] = 0

        data = data[np.argsort(np.sum(data, axis=1)),:]
        tot = np.sum(data, axis=1)

        model = GaussianMixture(2).fit(np.log(tot).reshape(-1, 1))
        w1, w2 = model.weights_
        sigma1, sigma2  = np.sqrt(model.covariances_)
        mu1, mu2 = model.means_

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
        
        self.datad = datad
        self.dataw = dataw
        self.pd = pd

        self.data0 = data
        self.tot0 = tot

        print('dry samples = {}, wet samples = {}'.format(idx, 39-idx))

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

class mc_sampler2:
    def __init__(self, thres=1, data=np.zeros([39,90])) -> None:
        self.thres = thres
        data[np.where(data<=self.thres)] = 0
        self.data = data[np.argsort(np.sum(data, axis=1)),:]
        self.T, self.N = self.data.shape
        pause = 1
        return

    def get_para(self):
        pp = [[0]*2 for _ in range(2)]
        for i in range(self.T):
            for j in range(2, self.N):
                f1 = 0 if self.data[i, j-1] <= self.thres else 1
                f2 = 0 if self.data[i, j] <= self.thres else 1
                pp[f1][f2] += 1
        p0 = pp[0][0] + pp[0][1]
        p1 = pp[1][0] + pp[1][1]
        pp[0][0], pp[0][1] = pp[0][0]/p0, pp[0][1]/p0
        pp[1][0], pp[1][1] = pp[1][0]/p1, pp[1][1]/p1

        self.prcp = self.data[np.where(self.data > self.thres)]

        self.pp = pp 
        self.p0 = p0/(p0+p1)
        self.p1 = p1/(p0+p1)
        return 
    
    def generate(self, dt=90, ic=[]):
        traj = np.zeros([dt+1,])
        if len(ic) == 0:
            d = np.random.rand()
            if d <= self.p1:
                traj[0] = np.random.choice(self.prcp)
        else:
            traj[0] = ic[0]
        for i in range(1, len(traj)):
            f1 = 0 if traj[i-1] == 0 else 1
            
            d = np.random.rand()
            if d <= self.pp[f1][1]:
                traj[i] = np.random.choice(self.prcp)

        return traj[1:]

if __name__ == '__main__':
    test = data_loader()
    pause = 1