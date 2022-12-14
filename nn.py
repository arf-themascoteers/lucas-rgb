import torch
import train
import test
import ds_manager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_r2(train_ds, test_ds):
    model = train.train(device, train_ds)
    r2 = test.test(device, model, test_ds)
    print(r2)
    return r2


def run_kfolds():
    dm = ds_manager.DSManager()
    r2s = []
    for train_ds, test_ds in dm.get_kfolds():
        r2s.append(get_r2(train_ds, test_ds))
    print(sum(r2s)/len(r2s))


def run_single():
    model = train.train(device)
    r2 = test.test(device, model)
    print(r2)


if __name__ == "__main__":
    run_single()