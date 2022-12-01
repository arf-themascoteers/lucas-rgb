import ds_manager
import numpy as np
import linear
from functools import partial

NP_FILE = "nps/single.npy"
SIZE = 3


def si(mat, x, y):
    return ((mat[:, x] - mat[:, y]) / (mat[:, x] + mat[:, y])).reshape(-1, 1)


def single(mat, x):
    return (mat[:, x]).reshape(-1, 1)


def run_plz():
    matrix = np.zeros([3])
    np.save(NP_FILE, matrix)
    dm = ds_manager.DSManager()
    train_ds = dm.get_train_ds()
    test_ds = dm.get_test_ds()
    train_y = train_ds.get_y()
    test_y = test_ds.get_y()

    for i in range(matrix.shape[0]):
        si_fun = partial(single, x=i)
        train_x = train_ds.get_si(si_fun)
        test_x = test_ds.get_si(si_fun)
        matrix[i] = linear.get_r2(train_x, train_y, test_x, test_y)
        print(f"Done {i} among {3}")
        np.save(NP_FILE, matrix)

    print("done")

