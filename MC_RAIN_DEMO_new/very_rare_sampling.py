import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import bisect
from MC_SAMPLING import mc_sampler2
from demo2 import GEN, LDS_demo

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