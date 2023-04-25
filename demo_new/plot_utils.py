import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# class shaded_line:
#     def __init__(self) -> None:
        
#     def plot_(self, dest):
        

        
#         return 

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 5)
with sns.axes_style("darkgrid"):
    epochs = list(range(101))
    for i in range(5):
        meanst = np.array(means.ix[i].values[3:-1], dtype=np.float64)
        sdt = np.array(stds.ix[i].values[3:-1], dtype=np.float64)
        ax.plot(epochs, meanst, label=means.ix[i]["label"], c=clrs[i])
        ax.fill_between(epochs, meanst-sdt, meanst+sdt ,alpha=0.3, facecolor=clrs[i])
    ax.legend()
    ax.set_yscale('log')