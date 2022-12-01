import ds_manager
import linear


def soci(mat):
    blue = mat[:, 0]
    green = mat[:, 1]
    red = mat[:, 2]
    return ((blue)/(red*green)).reshape(-1, 1)


def run_plz():
    dm = ds_manager.DSManager()
    train_ds = dm.get_train_ds()
    test_ds = dm.get_test_ds()
    train_y = train_ds.get_y()
    test_y = test_ds.get_y()

    train_x = train_ds.get_si(soci)
    test_x = test_ds.get_si(soci)
    r2 = linear.get_r2(train_x, train_y, test_x, test_y)
    print(f"{r2}")
    print("done")


run_plz()