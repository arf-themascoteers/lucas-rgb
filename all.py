import performance
import plotme

NP_FILE = "nps/ndi.npy"


def ndi(mat, x, y):
    return ((mat[:, x] - mat[:, y]) / (mat[:, x] + mat[:, y])).reshape(-1, 1)


performance.run_plz(NP_FILE, ndi)
plotme.plot_please(NP_FILE)