import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

def plot_please():
    array = np.load("nps/matrix.npy")
    mask = np.zeros_like(array)
    mask[np.tril_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(array, mask=mask, vmax=.3, square=True, cmap="mako")
        ax.invert_yaxis()
    plt.show()
    print("done")