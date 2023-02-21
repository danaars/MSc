import os
import sys
import path
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split

#directory = path.Path(__file__).abspath()
#sys.path.append(directory.parent.parent)

from mnist_cnn import MNISTClassify

class mnistaudio(Dataset):
    def __init__(self, csv_file, data_dir, transform=None, width_cut=None):
        self.data_table = pd.read_csv(os.path.join(data_dir, csv_file))
        self.data_dir = data_dir

        self.max_w = self.data_table["width"].max()
        self.max_h = self.data_table["height"].max()
        
        if transform == "pad":
            self.transform = F.pad

    def __getitem__(self, idx):
        df = self.data_table.iloc[idx]
        filename = df["filename"]
        w = df["width"]
        h = df["height"]
        file_pth = os.path.join(self.data_dir, filename)
        tens = torch.from_numpy(np.loadtxt(file_pth)).unsqueeze(0)

        label_idx = int(df["label"])
        label = torch.zeros(10)
        label[label_idx] = 1

        if self.transform:
            w_diff = self.max_w - w
            h_diff = self.max_h - h

            lft_pad = int(np.ceil(w_diff/2))
            rgt_pad = int(np.floor(w_diff/2))
            top_pad = int(np.ceil(h_diff/2))
            btm_pad = int(np.floor(h_diff/2))

            p = (lft_pad, rgt_pad, top_pad, btm_pad)
            #tens = self.transform(tens, p).double()
            tens = self.transform(tens, p)

        return tens, label_idx
    
    def __len__(self):
        return len(self.data_table)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    datadir = "data"
    metacsv = "meta.csv"

    ds = mnistaudio(metacsv, datadir, "pad")
    l = len(ds)
    trainsize = int(0.80*l)
    testsize = int(l-trainsize)

    train, test = random_split(ds, [trainsize, testsize])
    #print(f"Length train set: {len(train)}")
    #print(f"Length test set: {len(test)}")

    trainloader = DataLoader(train, batch_size=16, shuffle=True, num_workers=1)
    testloader = DataLoader(test, batch_size=16, shuffle=True, num_workers=1)

    sample = torch.rand([1, ds.max_w, ds.max_h]).to(device)
    net = MNISTClassify(sample_input = sample,
                        device = device).double()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = net.parameters())


    print("Starting training loop.")
    epochs = 1
    statinterval = 100
    start_train_time = time.time()
    for e in range(epochs):
        print(f"\nEpoch {e+1}/{epochs}")

        epoch_loss = 0
        tmp_loss = 0
        epoch_test_acc = []
        epoch_train_acc = []

        t0 = time.time()
        for i, data in enumerate(trainloader):
            tens, label = data
            tens = tens.to(device)
            label = label.to(device)
        
            # Prediction
            pred = net(tens)
            print(pred)
            print(label)
            print(pred.shape)
            print(label.shape)

            # Calculate loss
            loss = criterion(pred, label)

            loss_it = loss.item()
            epoch_loss += loss_it
            tmp_loss += loss_it

            # Backward sweep, and step in param space
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)%statinterval == 0:
                print(f"[{i+1}/{len(train)}]\tAvg. loss: {tmp_loss/statinterval}")
                tmp_loss = 0

        t1 = time.time()
        print(f"Time for epoch: {t1-t0} seconds")
        print(f"Epoch avg. loss: {epoch_loss/len(train)}")

        with torch.no_grad():
            train_acc = 0
            for i, data in enumerate(trainloader):
                tens, label = data
                tens = tens.to(device)
                label = label.to(device)

                pred = net(tens)
                _, pred_class = torch.max(pred, 1)
                train_acc += (pred_class == label).sum().item()

            epoch_train_acc.append(train_acc/len(train))

            test_acc = 0
            for i, data in enumerate(testloader):
                tens, label = data
                tens = tens.to(device)
                label = label.to(device)

                pred = net(tens)
                _, pred_class = torch.max(pred, 1)
                test_acc += (pred_class == label).sum().item()

            epoch_test_acc.append(test_acc/len(test))

    np.savetxt("train_epoch.txt", np.array(epoch_train_acc))
    np.savetxt("test_epoch.txt", np.array(epoch_test_acc))
