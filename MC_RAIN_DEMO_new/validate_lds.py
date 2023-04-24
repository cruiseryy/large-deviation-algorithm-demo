import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import bisect
from MC_SAMPLING import mc_sampler2
from time import time
from demo2 import GEN, LDS_demo

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

wd = np.zeros([1000, 39])
for kk in range(100):
    test = LDS_demo(gen=gg, k=0, N=39)
    test.update()
    test.evalute()
    wd[kk, :] = sorted(7.0784*90 - test.res[:,1], reverse=True)
    pause = 1

wd0 = np.sum(data, axis=1)
wd0 = wd0[::-1]

fig, ax = plt.subplots()
alpha = 1 
beta = 1
for y in range(39):
    low = np.quantile(wd[:, y], 0.05)
    upp = np.quantile(wd[:, y], 0.95)
    ax.plot([(39+beta)/(y+alpha), (39+beta)/(y+alpha)], [low, upp], color='black')
    ax.scatter((39+beta)/(y+alpha), wd0[y], marker='D', color='red', label='empirical')
    ax.scatter((39+beta)/(y+alpha), np.mean(wd[:, y]), marker='o', color='black', label='MC')
    if y == 0:
        ax.legend()
ax.set_ylabel('total prcp (mm)')
ax.set_xlabel('return period (yr)')
ax.set_ylim([100, 1400]) 
ax.set_xscale(value="log")
fig.tight_layout()
# plt.show()
pause = 1