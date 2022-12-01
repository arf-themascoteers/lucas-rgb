import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

NP_FILE = "nps/single.npy"


def plot_please():
    array = np.load(NP_FILE)
    x_ticks = np.arange(len(array))
    xticklabels = ["Blue", "Green", "Red"]
    plt.xticks(x_ticks, xticklabels, rotation='vertical')
    plt.bar(xticklabels, array)
    plt.show()
    print(array)
    print("done")