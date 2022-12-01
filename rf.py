from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import ds_manager
import numpy as np


def get_r2(train_x, train_y, test_x, test_y):
    reg = RandomForestRegressor(max_depth=15, n_estimators=1000).fit(train_x,train_y)
    return reg.score(test_x, test_y)


if __name__ == "__main__":
    dm = ds_manager.DSManager()
    train_ds = dm.get_train_ds()
    test_ds = dm.get_test_ds()
    train_x = train_ds.get_x()
    train_y = train_ds.get_y()
    test_x = test_ds.get_x()
    test_y = test_ds.get_y()
    score = get_r2(train_x, train_y, test_x, test_y)
    print(score)
