from station_data_stat import mc_lagone
from twolag_mc import mc_lagtwo
from matplotlib import pyplot as plt
import numpy as np

mc1 = mc_lagone()
mc2 = mc_lagtwo()

np.random.seed(916354)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[12, 8])

titles = [['(a) lag=1', '(b) lag=1'], ['(c) lag=2', '(d) lag=2']]
for i in range(2):
    for j in range(2):
        if i == 0:
            wd, wd0 = mc1.mc_simulation(k=j)
        else:
            wd, wd0 = mc2.mc_simulation(k=j)
        K = j
        alpha = 1 
        beta = 1
        for y in range(39):
            low = np.quantile(wd[:, y], 0.05)
            upp = np.quantile(wd[:, y], 0.95)
            ax[i][j].plot([low, upp],[(y+alpha)/(39+beta), (y+alpha)/(39+beta)], color='black')
            ax[i][j].scatter(wd0[y], (y+alpha)/(39+beta), marker='D', color='red', label='empirical')
            ax[i][j].scatter(np.mean(wd[:, y]), (y+alpha)/(39+beta), marker='o', color='black', label='MC')
            if i == j == y == 0:
                ax[i][j].legend(loc='upper left')

        ax[i][j].set_title(titles[i][j])
        ax[i][j].set_xlabel('total prcp (mm)')
        ax[i][j].set_ylabel('quantiles')
        ax[i][j].set_xlim([200, 1400])
        
fig.tight_layout()
plt.show()
fig.savefig('MC_RAIN_DEMO_new/fig1.pdf')


pause = 1