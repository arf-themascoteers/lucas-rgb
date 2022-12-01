import torch
import train
import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    model = train.train(device)
    r2 = test.test(device, model)
    print(r2)