import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T
import pandas as pd
import numpy as np
#import lvm_read

from torch.utils.data import Dataset, DataLoader


class Pipesound(Dataset):
    
    def __init__(self, csv_file, data_dir, transform=None, label_transform=None, width_cut=None):
        self.data_table = pd.read_csv(os.path.join(data_dir,csv_file))

        if width_cut:
            self.wc = width_cut
            self.data_table = self.data_table[self.data_table["width"] <= width_cut]
            self.data_table = pd.concat([self.data_table[self.data_table["slug_label"]==1].head(5), 
                                         self.data_table[self.data_table["slug_label"]==0].head(5)])

        self.max_w = self.data_table["width"].max()
        self.max_h = self.data_table["height"].max()

        self.data_dir = data_dir

        if transform == "pad":
            self.transform = F.pad      # Padding punc

        self.label_transform = label_transform

    def __getitem__(self, idx):
        #print("Index: ", idx)
        #print("Table for given index\n", self.data_table.iloc[idx])
        df = self.data_table.iloc[idx]
        filename = df['filename']
        w = df["width"]
        h = df["height"]
        #print("Filename: ", filename)
        file_pth = os.path.join(self.data_dir, filename)
        #tens = torch.from_numpy(np.loadtxt(file_pth)).double()
        tens = torch.from_numpy(np.loadtxt(file_pth))
        label_idx = int(df['slug_label'])
        label = torch.zeros(2)
        label[label_idx] = 1
        #print("Label = ", label)

        if self.transform:
            w_diff = self.max_w - w
            h_diff = self.max_h - h
            #w_diff = self.wc - w
            #h_diff = self.max_h - h
            
            lft_pad = int(np.ceil(w_diff/2))
            rgt_pad = int(np.floor(w_diff/2)) 
            top_pad = int(np.ceil(h_diff/2))
            btm_pad = int(np.floor(h_diff/2))

            p = (lft_pad, rgt_pad, top_pad, btm_pad)
            #tens = self.transform(tens, p).double()
            tens = self.transform(tens, p)

        if self.label_transform:
            label = self.label_transform(label, dtype = torch.float64)

        return tens, label
    
    def __len__(self):
        return len(self.data_table)
    
    def getdatadims(self):
        return 


class Soundalyzer(nn.Module):

    def __init__(self, channels, sample_input):
        super(Soundalyzer, self).__init__()
        
        """
        Image; 3 channels (mic tapes) #, of size width, height.
        This combination of convolutions and maxpooling could be optimized,
        but this setup is common for image classification.
        """

        self.conv1 = nn.Conv2d(channels, 8, 5)     # 3 channels in, 8 filters applied pr. channel, 5x5 size pr. filter
        # out: 3x8
        self.conv2 = nn.Conv2d(8, 16, 3)
        # out: 3x8x16
        self.pool = nn.MaxPool2d(2)           # Maxpool over 2x2 squares

        x = self.pool(F.relu(self.conv1(sample_input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x)
        
        flatten_len = len(x)

        # Classification layers
        #self.classify1 = nn.Linear(127200, 64)
        self.classify1 = nn.Linear(flatten_len, 64)
        self.classify2 = nn.Linear(64, 64)
        self.classify3 = nn.Linear(64, 2)

        # Velocity regression layers
        #self.vel1 = nn.Linear(542880, 5)
        self.vel1 = nn.Linear(flatten_len, 5)
        self.vel2 = nn.Linear(5, 5)
        self.vel3 = nn.Linear(5, 1)

        # Length regression layers
        #self.len1 = nn.Linear(542880, 5)
        self.len1 = nn.Linear(flatten_len, 5)
        self.len2 = nn.Linear(5, 5)
        self.len3 = nn.Linear(5, 1)

    def forward(self, data):

        """ Apply convolutions + pooling, and flatten result. """
        x = self.pool(F.relu(self.conv1(data)))
        #print(x.shape)
        #print(x)
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        #print(x)
        x = torch.flatten(x)
        #print(x.shape)
        #print(x)
        
        """ Network for image classification """
        pred_class = F.relu(self.classify1(x))
        #print(pred_class.shape)
        #print(pred_class)
        pred_class = F.relu(self.classify2(pred_class))
        #print(pred_class.shape)
        #print(pred_class)

        #pred_class = F.log_softmax(self.classify3(pred_class), dim = 0) # Softmaxed output - "probabilities"
        x = self.classify3(pred_class)
        m = nn.Sigmoid()
        pred_class = x

        #pred_class = self.classify3(pred_class) # Softmaxed output - "probabilities"
        #print(pred_class.shape)
        #print(pred_class)
        #pred_class = self.classify3(pred_class) # Linear test


        '''
        """ Network for velocity regression """
        # ReLU vs Sigmoid for this one?
        pred_vel = F.relu(self.vel1(x))
        pred_vel = F.relu(self.vel2(pred_vel))
        pred_vel = self.vel3(pred_vel)              # Linear output, no restriction on the velocities

        """ Network for slug length regression """
        # ReLU vs Sigmoid for this one?
        pred_len = F.relu(self.len1(x))
        pred_len = F.relu(self.len2(pred_len))
        pred_len = self.vel3(pred_vel)              # Linear output
        '''

        #return pred_class, pred_vel, pred_len
        return pred_class

    #def loss(self, pred, label_idx):
    def loss(self, pred, label):
        #pred_class, pred_vel, pred_len = pred
        #true_class, true_vel, true_len = label
        pred_class = pred
        #true_class = torch.zeros(2)
        #true_class[label_idx] = 1
        true_class = label

        classification_loss = nn.CrossEntropyLoss()
        #vel_regression_loss = nn.MSELoss(pred_vel, true_vel)
        #len_regression_loss = nn.MSELoss(pred_len, true_len)
        

        #tot_loss = classification_loss# + vel_regression_loss + len_regression_loss
        #print(pred_class, true_class)
        #print(pred_class.shape, true_class.shape)
        tot_loss = classification_loss(pred_class, true_class)

        return tot_loss
    
    def train(self, dataset, optim, epochs=1000, batch_size=1, shuffle=True, num_workers=1):

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        self.loss_epoch = []
        self.train_acc = []
        self.test_acc = []
        for nr, epoch in enumerate(range(epochs)):
            print(f"Epoch {nr + 1}/{epochs}")
            epoch_loss = 0
            t0 = time.time()
            #before_params = self.parameters()
            #before_params2 = self.parameters()
            #print("Before == before: ", before_params == before_params2)
            for i, data in enumerate(dataloader):
                tens, label = data
                #epoch_loss = 0
                #print(tens, label)
                #print(label.item())
                print("Input shape: ", tens.shape)

                # Set param-space gradient to zero
                optim.zero_grad()

                # Get model output
                #print("into self")
                #print("Inserting tens data into model")
                out = self(tens)
                print("Pred: ", out)
                #print(type(out))
                print(out.shape)
                print("Target: ", label[0])
                #print(type(label[0]))
                print(label[0].shape)
                #print("self call successful")
                # Calculate loss
                #print("into loss")
                l = self.loss(out, label[0])
                #print(out)
                #print(label)
                #print(l)
                #print("")
                #print("loss: ", l.item())
                l.backward()
                epoch_loss += l.item()
                #print("loss call successful")
                # Perform backpropagation
                #l.backward()
                #print("backprop successful")
                # Step in param space
                optim.step()
                #print("optim successful")

            t1 = time.time()
            print("Epoch time: ", t1-t0)
            self.loss_epoch.append(epoch_loss/len(dataset))
            print("last avg. loss: ", self.loss_epoch[-1])
            #after_params = self.parameters()
            #print("Equal parameters: ", before_params == after_params)

            
            with torch.no_grad():
                m = nn.Sigmoid()
                n = nn.Softmax()
                hit = 0
                for data in dataloader:
                    tens, label = data
                    #print(self(tens)[0])
                    pred = n(self(tens))
                    
                    predind = pred.argmax().item()
                    labelind = label[0].argmax().item()

                    print(f"PRED: {pred}\tIND: {predind}")
                    print(f"LABEL: {label[0]}\tIND: {labelind}")
                    if predind == labelind:
                        hit += 1
                acc = hit/len(dataset)
            print("Accuracy (training data): ", acc)

        self.loss_epoch = np.array(self.loss_epoch)

        print("Training successful")

    #def load_model(self, modelfile):
                
    def predict(self, data):
        with torch.no_grad():
            return self(data)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    data_dir = "0811_spectro_scaled"
    #dataset = Pipesound(os.path.join(data_dir, "meta.csv"), data_dir, label_transform=torch.tensor)
    dataset = Pipesound(os.path.join(data_dir, "meta.csv"), data_dir) 

    model = Soundalyzer()
    model = model.double()

    optimizer = optim.Adam(params = model.parameters(), lr = 0.1)
    e = 50

    print("Starting training")
    model.train(dataset, optimizer, epochs = e, num_workers = 2)
    print("Finished training")

    plt.plot(np.arange(1, e+1), model.loss_epoch)
    plt.show()
