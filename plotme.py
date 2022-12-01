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
        ax = sns.heatmap(array, mask=mask, cmap="mako", annot=True, fmt='.3f')
        ax.invert_yaxis()
        plt.xticks(x_ticks, xticklabels, rotation='horizontal', va="top", ha="left" )
        plt.yticks(x_ticks, xticklabels, rotation='horizontal', va = "center")
    plt.show()
    print(array)
    print("done")

if __name__ == "__main__":
    NP_FILE = "nps/ndi.npy"
    plot_please(NP_FILE)