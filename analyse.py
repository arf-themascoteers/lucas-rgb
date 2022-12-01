import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

def transform_index(ind):
    return ind * 50 + 400

def get_bands(index):
    unraveled = np.unravel_index(index, (40, 40))
    return transform_index(unraveled[0]), transform_index(unraveled[1])

def anal():
    array = np.load("nps/matrix.npy").flatten()
    indices = np.argsort(array)
    for i in range(10):
        index = indices[len(indices) - 1 - i]
        b1, b2 = get_bands(index)
        print(b1, b2, index)

anal()