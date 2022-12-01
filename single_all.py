import single_performance
import single_plotme

NP_FILE = "nps/inv_cube.npy"


def single(mat, x):
    return (1/(mat[:, x]**3)).reshape(-1, 1)


single_performance.run_plz(NP_FILE, single)
single_plotme.plot_please(NP_FILE)