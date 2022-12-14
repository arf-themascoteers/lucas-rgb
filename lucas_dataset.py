import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import colorsys


class LucasDataset(Dataset):
    def __init__(self, source):
        self.scaler = None
        self.df = self._preprocess(source)
        self.x = self.df[:,1:]
        self.hsv = np.zeros_like(self.x)
        for index, row in enumerate(self.x):
            a_hsv = colorsys.rgb_to_hsv(row[2], row[1], row[0])
            self.hsv[index, 0] = a_hsv[0]
            self.hsv[index, 1] = a_hsv[1]
            self.hsv[index, 2] = a_hsv[2]
        # blue = self.x[:, 0]
        # green = self.x[:, 1]
        # red = self.x[:, 2]
        #soci = ((blue) / (red * green)).reshape(-1, 1)
        # inv_green = (1/green).reshape(-1, 1)
        # inv_blue_sq = (1/(blue**2)).reshape(-1, 1)
        self.x = np.concatenate((self.x, self.hsv), axis=1)


    def _preprocess(self, source):
        self.scaler = MinMaxScaler()
        x_scaled = self.scaler.fit_transform(source[:, 0].reshape(-1, 1))
        source[:, 0] = np.squeeze(x_scaled)
        source[:, 1:] = 1 / (10 ** source[:, 1:])
        return source

    def unscale(self, value):
        return self.scaler.inverse_transform([[value]])[0][0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        this_x = self.get_x()[idx]
        soc = self.get_y()[idx]
        return torch.tensor(this_x, dtype=torch.float32), torch.tensor(soc, dtype=torch.float32)

    def get_y(self):
        return self.df[:,0]

    def get_x(self):
        return self.x
        #return

    def get_si(self, sif):
        return sif(self.get_x())


