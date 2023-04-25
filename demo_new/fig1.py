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
tmpres = np.zeros([100, 10000])

for kk in range(100):
    t1 = time()
    tmp_run = LDS_demo(gen=gg, k=0, N=10000)
    tmp_run.update()
    tmp_run.evalute()
    tmpres[kk, :] = thres0*ta - tmp_run.res[:, 1]
    t2 = time()
    print('iteration {0}, {1:0.4f} for this iteration'.format(kk, t2 - t1))
    t1 = time()

alter_run = LDS_demo(gen=gg, k=0.02, N=128)
alter_run.update()
alter_run.evalute()

alter_run1 = LDS_demo(gen=gg, k=0.001, N=128)
alter_run1.update()
alter_run1.evalute()

freq0 = [(i+1)/39 for i in range(39)]
freq0 = np.array(freq0)
dl.tot0.sort()
tot = thres0*ta - dl.tot0

rare_tot = [127.86, 182.78, 243.06]
rare_freq = [1.0/10000, 1.0/1000, 1.0/100]
freq = tmp_run.res[:, 0]
fig, ax = plt.subplots(figsize=[8, 4.5])
with sns.axes_style("darkgrid"):
    plt.grid(True, which='both')
    med = np.median(tmpres, axis=0)
    upp = np.quantile(tmpres, 0.99, axis=0)
    low = np.quantile(tmpres, 0.01, axis=0)
    ax.plot(1/freq, med, color='navy', label='median')
    ax.fill_between(1/freq, low, upp, alpha=0.3, facecolor='navy', label='1-99 precentiles')

    ax.scatter(1/alter_run.res[:,0], thres0*ta - alter_run.res[:,1], s=50, color='mediumseagreen', marker='P', label='mc_k=0.02')
    ax.scatter(1/alter_run1.res[:,0], thres0*ta - alter_run1.res[:,1], s=50, color='darkgreen', marker='o', label='mc_k=0.001')
    ax.scatter(1/freq0, thres0*ta - tot, marker='D', s=30, color='black', label='obs')
    for ri in range(3):
        if ri == 0:
            ax.scatter(1/rare_freq[ri], rare_tot[ri], marker='*', s=100, color='darkblue', label='Rare Events')
        else:
            ax.scatter(1/rare_freq[ri], rare_tot[ri], marker='*', s=100, color='darkblue')

    ax.set_xscale('log')
    ax.set_xlabel('return period (yr)')
    ax.set_ylabel('total rainfall (mm)')
    ax.legend()
    ax.set_xlim([0.7, 3e4])
    ax.set_ylim([0, 1500])
    fig.savefig('demo_new/fig1.pdf')

pause = 1
