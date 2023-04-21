import numpy as np 
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import bisect
from MC_SAMPLING import mc_sampler, mc_sampler2
from time import time
from copy import deepcopy

np.random.seed(7)

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
# tmpmax = deepcopy(data[-1, :])
# data[-1, :] = data[-2, :]
tot = np.sum(data, axis=1)
kde = stats.gaussian_kde(np.log(tot))
# xx = np.linspace(np.log(tot).min(), np.log(tot).max(), 100)
# pp = kde(xx)

model = GaussianMixture(2).fit(np.log(tot).reshape(-1, 1))
w1, w2 = model.weights_
sigma1, sigma2  = np.sqrt(model.covariances_)
mu1, mu2 = model.means_

# pp1 = 1/np.sqrt(2*np.pi) /sigma1 * np.exp(-(xx - mu1)**2/2/sigma1**2)
# pp2 = 1/np.sqrt(2*np.pi) /sigma2 * np.exp(-(xx - mu2)**2/2/sigma2**2)
txx = np.linspace(6, 6.4, 10000)
ppx1 = 1/np.sqrt(2*np.pi) /sigma1 * np.exp(-(txx - mu1)**2/2/sigma1**2) * w1
ppx2 = 1/np.sqrt(2*np.pi) /sigma2 * np.exp(-(txx - mu2)**2/2/sigma2**2) * w2
diff = np.abs(ppx1 - ppx2).reshape(-1)
dvd0 = txx[np.where(diff == np.min(diff))]

# ppm = w1*pp1 + w2*pp2
# fig, ax = plt.subplots()
# ax.plot(xx, pp)
# ax.plot(xx, ppm.reshape(-1, 1))
# pause = 1

dvd = np.exp(dvd0)
dry = np.sum(tot <= dvd)
wet = np.sum(tot > dvd)

# pd, pw = dry/(dry+wet), wet/(dry+wet)
pd, pw = min(w1, w2), max(w1, w2)
idx = bisect.bisect(tot, dvd)
# idx = 11
# pd = idx/39
# pw = 1 - pd

datad = data[:idx, :]
dataw = data[idx:, ]

mc1 = mc_sampler2(thres=1, data=datad)
mc1.get_para()
mc2 = mc_sampler2(thres=1, data=dataw)
mc2.get_para()

# n = 1000
# tmp_tot = np.zeros([n,])
# for i in range(n):
#     t1 = time()
#     d = np.random.rand()
#     if d <= pd:
#         tmp_traj = mc1.generate()
#     else:
#         tmp_traj = mc2.generate()
#     tmp_tot[i] = np.sum(tmp_traj)
#     t2 = time()
#     print((i, t2-t1))


# kde2 = stats.gaussian_kde(tmp_tot)
# # xx2 = np.linspace(tmp_tot.min(), tmp_tot.max(), 100)
# pp2 = kde2(xx)
# ax.plot(xx, pp2)

nn = 1000
wd = np.zeros([nn, 39])
for k in range(nn):
    t1 = time()
    tmp_tot = np.zeros([39, ])
    for y in range(39):
        d = np.random.rand()
        if d <= pd:
            tmp_traj = mc1.generate()
        else:
            tmp_traj = mc2.generate()
        tmp_tot[y] = np.sum(tmp_traj)
    tmp_tot.sort()
    pause = 1
    wd[k, :] = tmp_tot
    t2 = time()
    print((k, t2 - t1))

# data[-1, :] = tmpmax
wd0 = np.sum(data, axis=1)
fig, ax = plt.subplots()
alpha = 1 
beta = 1
for y in range(39):
    low = np.quantile(wd[:, y], 0.05)
    upp = np.quantile(wd[:, y], 0.95)
    ax.plot([low, upp],[(y+alpha)/(39+beta), (y+alpha)/(39+beta)], color='black')
    ax.scatter(wd0[y], (y+alpha)/(39+beta), marker='D', color='red', label='empirical')
    ax.scatter(np.mean(wd[:, y]), (y+alpha)/(39+beta), marker='o', color='black', label='MC')
    if y == 0:
        ax.legend()
ax.set_xlabel('total prcp (mm)')
ax.set_ylabel('quantiles')
ax.set_xlim([100, 1400]) 

fig.tight_layout()
# plt.show()
# pause = 1
fig.savefig('MC_RAIN_DEMO_new/fig3_lag1mc.pdf')
pause = 1