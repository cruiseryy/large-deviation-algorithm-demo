import numpy as np
import xarray as xr
import glob
import os
import bisect

class LDS:
    def __init__(self, 
                 path = './',
                 pre = 'wrfrst', 
                 wrf_wd = './wrfout/',
                 ref = 10, 
                 N = 10, 
                 K = 1, 
                 T = 18, 
                 dt = 5) -> None:
        # paths of the working directory and wrf direcotory
        # prefix of the wrfrst/traj files
        self.path = path 
        self.pre = pre
        self.wrf_wd = wrf_wd
        # a rescaling coefficient for the weights
        self.K = K 
        # number of trajectories
        self.N = N
        # to record the current timestep
        self.timer = 0
        # all traj file paths are renamed and stored 
        file_ls = sorted(glob.glob(self.path + self.pre + '*'))
        for idx, val in enumerate(file_ls):
            os.rename(val, 'traj' + str(idx) +'_'+str(self.timer))
        # dt = len of each sub-interval
        # T = number of dts
        self.T = T
        self.dt = dt
        # to store intermediate variables 
        self.weights = np.zeros([self.T, self.N])
        self.R = np.zeros([self.T,])
        # goto[i][j0] = j1, the j0 traj at timer i goes to j1 traj at timer i+1
        # comefrom[i+1][j1] = j0, the j1 traj at timer i+1 comes from j0 at timer i
        self.comefrom = {{} for _ in range(self.N)}
        # baseline value per unit time for computing deficit
        self.ref = ref
        # replace with traj_j_prev and traj_j_cur if do not want to save intermediate info
        pause = 1
    
    def run(self) -> None:
        if self.timer < self.T:
            self.timer += 1
            self.update(self.timer)
            self.eval(self.timer)
            self.resample(self.timer)
            self.perturb(self.timer)
        return

    def update(self, i) -> None:
        for j in range(self.N):
            tmp_ic = 'traj' + str(j) + '_' + str(i-1)
            # run wrf here with the given ICs
            cmd = 'echo "pretend we are running wrf here at time step {0}"'.format(i)
            os.system(cmd)
            # this needs to be changed, these three lines for renaming the wrfout/wrfrst files
            wrf_output = self.wrf_wd + self.pre + str(j)
            dest = self.path + 'traj' + str(j) +'_'+str(i)
            os.rename(wrf_output, dest)
        return

    def eval(self, i) -> None:
        for j in range(self.N):
            # since wrfout/wrfrst saves accumulated precipitation, wrfrst at the previous timestep (i.e., IC)
            # needs to be used for computing precipitation deficit 
            prev = xr.open_dataset(self.path + 'traj' + str(j) + '_' + str(i-1))
            cur = xr.open_dataset(self.path + 'traj' + str(j) + '_' + str(i))
            tmp_prcp = np.sum(cur['RAINNC'] - prev['RAINNC'])
            # for now, i assign the weights proportional to exp(precip deficit)
            self.weights[i,j] = np.exp(self.k*(self.ref*self.dt - tmp_prcp))
        # R = mean(weights)
        self.R[i] = np.mean(self.weights[i,:])
        # rescaling weights
        self.weights[i,:] /= self.R[i]
        return
    
    def resample(self, i) -> None:
        # the cdf estimated using weights is used to draw new trajs
        tmpcdf = np.zeros([self.N,])
        tmpcdf[0] = self.weights[i,0]
        for j in range(1, self.N):
            tmpcdf[j] = tmpcdf[j-1] + self.weights[i,j]
        tmpcdf /= tmpcdf[-1]
        # repeat the sampling for N times
        for j in range(self.N):
            idx = bisect.bisect(tmpcdf, np.random.rand())
            self.comefrom[i,j] = idx
            # copy potential new trajs as temp trajs
            # this is to not ruin the original trajs before resampling all new trajs
            source = 'traj' + str(idx) + '_' + str(i)
            dest = 'tmp_traj' + str(j) + '_' + str(i)
            cmd = 'cp ' + source + ' ' + dest
            os.system(cmd)
        # now the resampling is finished, some of the original trajs can be discarded
        for j in range(self.N):
            source = 'tmp_traj' + str(j) + '_' + str(i)
            dest = 'traj' + str(j) + '_' + str(i)
            os.rename(source, dest)
        return
    
    def perturb(self, i):
        for j in range(self.N):
            pause = 1
        return 

if __name__ == "__main__":

    pause = 1
    sampler = LDS(pre = 'traj', K = 1e-3)
    sampler.run()
