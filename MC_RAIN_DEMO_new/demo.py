import numpy as np
import bisect
from matplotlib import pyplot as plt

class ToyModel:
    def __init__(self, P, MU = 2, SIGMA = 0.5, T = 200, K = 0, DT = 10, N = 128):
        self.p = P
        self.p_inf = (P**50)[:, 0]
        self.mu = MU
        self.sigma = SIGMA
        self.Ta = T
        self.k = K
        self.dt = DT
        self.n = N
        self.traj = np.zeros([T, N])
        self.R = np.zeros(T//DT)

        self.res = np.zeros([N, 2])
        self.intensity = np.zeros(N)

    def evalute(self):
        self.intensity = [(self.score(self.traj[:,j]), j) for j in range(self.n)]
        self.intensity.sort(key = lambda x: x[0], reverse=True)
        dp = 1.0 / self.n
        if self.k != 0:
            lambda_ = np.sum(np.log(self.R))
            alter_p = [np.exp(-self.k * self.intensity[i][0])*np.exp(lambda_) for i in range(self.n)]
        else: 
            alter_p = [1]*self.n
        self.res[0, :] = [alter_p[0]*dp, self.intensity[0][0]]
        for j in range(1, self.n):
            self.res[j, :] = [self.res[j-1,0] + dp*alter_p[j], self.intensity[j][0]]
        return

    def update(self):
        if self.k == 0:
            for j in range(self.n):
                self.traj[:,j] = self.mc(self.Ta, seed = j)
        else: 
            tmp_traj = np.zeros([self.dt, self.n])
            for i in range(self.Ta // self.dt):
                for j in range(self.n):
                    if i == 0: 
                        ic = -1
                    else:
                        ic = 0 if new_traj[-1, j] == 0 else 1
                    tmp_traj[:, j] = self.mc(self.dt, ic)
                new_traj, tmpr = self.restart(tmp_traj)
                self.R[i] = tmpr
                self.traj[i*self.dt: (i+1)*self.dt, :] = new_traj
        return

    def mc(self, t, ic = -1, seed = 0): 
        # np.random.seed(seed)
        cur = 0
        traj = np.zeros(t)
        if ic == -1:
            ic = 0 if np.random.rand() <= self.p_inf[0] else 1
        while cur < t:
            if ic == 0:
                ic = 0 if np.random.rand() <= self.p[0, 0] else 1
            else:
                traj[cur] = np.exp(np.random.normal(self.mu, self.sigma))
                ic = 0 if np.random.rand() <= self.p[1, 0] else 1
            cur += 1
        return traj
    
    
    def restart(self, traj):
        n, m = len(traj), len(traj[0])
        weights = np.zeros(m)
        new_traj = np.zeros([n, m])

        for j in range(m):
            weights[j] = np.exp(self.k * self.score(traj[:,j]))
        R = np.mean(weights)
        weights /= R

        tmpcdf = np.zeros(m)
        tmpcdf[0] = weights[0]
        for j in range(1, m):
            tmpcdf[j] = tmpcdf[j-1] + weights[j]
        tmpcdf /= tmpcdf[-1]

        for j in range(m):
            new_traj[:, j] = traj[:, bisect.bisect(tmpcdf, np.random.rand())]

        return new_traj, R
    
    def score(self, traj):
        n = len(traj)
        thres = 2.765
        return (thres*n - np.sum(traj))

# def tmplot(run1, run2):
#     return



if __name__ == "__main__":
    p = np.matrix([[3/4, 1/4], [1/2, 1/2]])
    control_run = ToyModel(P = p, N = 100000)
    control_run.update()
    control_run.evalute()
    alter_run = ToyModel(P = p, K = 0.05)
    alter_run.update()
    alter_run.evalute()
    plt.figure()
    plt.plot(1/control_run.res[:,0], control_run.res[:,1])
    plt.scatter(1/alter_run.res[:,0], alter_run.res[:,1])
    plt.xscale('log')

    pause = 1