import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


def plot_please(NP_FILE):
    array = np.load(NP_FILE)
    x_ticks = np.arange(len(array))
    xticklabels = ["Blue", "Green", "Red"]
    plt.xticks(x_ticks, xticklabels, rotation='vertical')
    plt.bar(xticklabels, array)
    for index, value in enumerate(array):
        plt.text(index, value, f"{value:.3f}")
    plt.show()
    print(array)
    print("done")