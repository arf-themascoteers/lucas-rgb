import performance
import plotme
import numpy as np


def ndi(mat, x, y):
    return ((mat[:, x] - mat[:, y]) / (mat[:, x] + mat[:, y])).reshape(-1, 1)


def di(mat, x, y):
    return ((mat[:, x] - mat[:, y])).reshape(-1, 1)


def ri(mat, x, y):
    return ((mat[:, x] / mat[:, y])).reshape(-1, 1)


def rdvi(mat, x, y):
    return ((mat[:, x] - mat[:, y]) / np.sqrt(mat[:, x] + mat[:, y])).reshape(-1, 1)


def msr(mat, x, y):
    return (((mat[:, x] / mat[:, y]) -1) / ((mat[:, x] / mat[:, y]) +1)).reshape(-1, 1)


def ari(mat, x, y):
    i = mat[:, x]
    j = mat[:, y]
    return ( (np.abs((i**2)-(j**2)))/(np.sqrt(i+j)) ).reshape(-1, 1)


def dri(mat, x, y):
    i = mat[:, x]
    j = mat[:, y]
    return (np.log10(i/j)).reshape(-1, 1)


def dsr(mat, x, y):
    i = 1/mat[:, x]
    j = 1/mat[:, y]
    return (np.log10(i) / np.log10(j)).reshape(-1, 1)


def dani(mat, x, y):
    i = np.log10(1/mat[:, x])
    j = np.log10(1/mat[:, y])
    return ((i-j)/(i+j)).reshape(-1, 1)


if __name__ == "__main__":
    name = "dani"
    NP_FILE = f"nps/{name}.npy"
    fun = vars()[name]
    performance.run_plz(NP_FILE, fun)
    plotme.plot_please(NP_FILE)

