import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


def plot_please(NP_FILE):
    array = np.load(NP_FILE)
    mask = np.zeros_like(array)
    mask[np.tril_indices_from(mask)] = True
    x_ticks = [0,1,2]
    xticklabels = ["Blue", "Green", "Red"]
    with sns.axes_style("white"):
        ax = sns.heatmap(array, mask=mask, square=True, cmap="mako")
        ax.invert_yaxis()
        plt.xticks(x_ticks, xticklabels, rotation='vertical', va="top" )
        plt.yticks(x_ticks, xticklabels, rotation='horizontal')
    plt.show()
    print(array)
    print("done")