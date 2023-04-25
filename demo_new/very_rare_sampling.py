import numpy as np
from matplotlib import pyplot as plt
from utils import data_loader, GEN, mc_sampler2
from demo2 import LDS_demo
from time import time
import seaborn as sns

np.random.seed(777)
thres0 = 7.07842
ta = 90
dl = data_loader()
# fit the two MC generators 
mc1 = mc_sampler2(thres=1, data=dl.datad)
mc1.get_para()
mc2 = mc_sampler2(thres=1, data=dl.dataw)
mc2.get_para()
# to make things less messy, i used one GEN to call both MC generators
gg = GEN(mc1, mc2, dl.pd)

ref0 = LDS_demo(gen=gg, k=0, N=10000)
ref0.update()
ref0.evalute()

ref_tot = [(thres0*ta -ref0.score(ref0.traj[j,:]), j) for j in range(ref0.N)]
ref_tot.sort(key = lambda x: x[0])
idx = [0, 10, 100]
for i in idx:
    print(ref_tot[i])

nn = 100
alter_res = np.zeros([2*4*nn, 128])
cur = 0
for kk in range(nn):
    alter_run = LDS_demo(gen=gg, k=0.02, N=128)
    alter_run.update()
    alter_run.evalute()
    alter_res[cur+2*kk, :] = alter_run.res[:, 0]
    alter_res[cur+2*kk+1, :] = thres0*ta - alter_run.res[:, 1]
cur += 2*nn

for i in range(3):
    
    print('\n ----------------------')
    print(mc1.pp)
    print((mc1.p0, mc1.p1))
    print('---------------------- \n')

    tmpidx = ref_tot[idx[i]][1]
    dl.datad[0, :] = ref0.traj[tmpidx, :]
    mc1 = mc_sampler2(thres=1, data=dl.datad)
    mc1.get_para()
    gg = GEN(mc1, mc2, dl.pd)

    for kk in range(nn):
        alter_run = LDS_demo(gen=gg, k=0.02, N=128)
        alter_run.update()
        alter_run.evalute()
        alter_res[cur+2*kk, :] = alter_run.res[:, 0]
        alter_res[cur+2*kk+1, :] = thres0*ta - alter_run.res[:, 1]
    cur += 2*nn

    pause = 1

# print(mc1.pp)
# print((mc1.p0, mc1.p1))
color_ = ['black', 'lightskyblue', 'cornflowerblue', 'navy']
marker_ = ['>', 'H', 'p', 'D']
label_ = ['base', '1/10000', '1/1000', '1/100']
xx = np.array([100+i*100 for i in range(10)])

fig, ax = plt.subplots(figsize=[8,4.5])

cur = 0
cur += 2*nn
for i in range(1, 4):
    tmpres = np.zeros([nn, 10])
    for kk in range(nn):
        tmp_return = 1/alter_res[cur+2*kk, :] 
        tmp_int = alter_res[cur+2*kk+1, :] 
        tmpres[kk, :] = np.interp(xx, tmp_return[::-1], tmp_int[::-1])
    txx = xx + 30 - (i-1)*20
    ax.scatter(txx, np.median(tmpres, axis=0), marker=marker_[i], color=color_[i], s=40, label=label_[i])
    for jdx, j in enumerate(txx):
        ax.plot([j, j], [np.quantile(tmpres[:, jdx], 0.25), np.quantile(tmpres[:, jdx], 0.75)], color=color_[i])

    cur += 2*nn 

cur = i = 0
tmpres = np.zeros([nn, 10])
for kk in range(nn):
    tmp_return = 1/alter_res[cur+2*kk, :] 
    tmp_int = alter_res[cur+2*kk+1, :] 
    tmpres[kk, :] = np.interp(xx, tmp_return[::-1], tmp_int[::-1])
txx = xx - 30
ax.scatter(txx, np.median(tmpres, axis=0), marker=marker_[i], color=color_[i], s=40, label=label_[i])
for jdx, j in enumerate(txx):
    ax.plot([j, j], [np.quantile(tmpres[:, jdx], 0.25), np.quantile(tmpres[:, jdx], 0.75)], color=color_[i])
    
ax.set_xticks(xx)
ax.set_xlabel('return period (yr)')
ax.set_ylabel('total rainfall (mm)')
ax.legend()
plt.grid(True, which='both')
fig.savefig('demo_new/fig2.pdf')
pause = 1