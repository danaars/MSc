import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import lvm_read

from torch.utils.data import Dataset, DataLoader

class Pipesound(Dataset):
    
    def __init__(self, csv_file, data_dir, transform=None, label_transform=None):
        self.data_table = pd.read_csv(csv_file)

        self.data_dir = data_dir
        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, idx):
        file_pth = os.path.join(self.data_dir, self.data_table.iloc[idx]['name'])
        tens = torch.load(file_pth)
        label_idx = int(self.data_table.iloc[idx]['slug'])
        #label = torch.zeros(2)
        #label[label_idx] = 1
        #print("Label = ", label)

        """
        if self.transform:
            tens = self.transform(tens)

        if self.label_transform:
            label = self.label_transform(label, dtype = torch.float64)
        """

        return tens, label_idx
    
    def __len__(self):
        return len(self.data_table)


class Soundalyzer(nn.Module):

    def __init__(self):
        super(Soundalyzer, self).__init__()
        
        """
        Image; 3 channels (mic tapes) #, of size width, height.
        This combination of convolutions and maxpooling could be optimized,
        but this setup is common for image classification.
        """

        self.conv1 = nn.Conv2d(3, 8, 5)     # 3 channels in, 8 filters applied pr. channel, 5x5 size pr. filter
        # out: 3x8
        self.conv2 = nn.Conv2d(8, 16, 3)
        # out: 3x8x16
        self.pool = nn.MaxPool2d(2)           # Maxpool over 2x2 squares

        # Classification layers
        self.classify1 = nn.Linear(127200, 64)
        self.classify2 = nn.Linear(64, 64)
        self.classify3 = nn.Linear(64, 2)

        # Velocity regression layers
        self.vel1 = nn.Linear(542880, 5)
        self.vel2 = nn.Linear(5, 5)
        self.vel3 = nn.Linear(5, 1)

        # Length regression layers
        self.len1 = nn.Linear(542880, 5)
        self.len2 = nn.Linear(5, 5)
        self.len3 = nn.Linear(5, 1)

    def forward(self, data):

        """ Apply convolutions + pooling, and flatten result. """
        x = self.pool(F.relu(self.conv1(data)))
        #print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = torch.flatten(x)
        #print(x.shape)
        
        """ Network for image classification """
        pred_class = F.relu(self.classify1(x))
        #print(pred_class.shape)
        pred_class = F.relu(self.classify2(pred_class))
        #print(pred_class.shape)
        pred_class = F.softmax(self.classify3(pred_class)) # Softmaxed output - "probabilities"
        #print(pred_class.shape)
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

    def loss(self, pred, label_idx):
        #pred_class, pred_vel, pred_len = pred
        #true_class, true_vel, true_len = label
        pred_class = pred
        true_class = torch.zeros(2)
        true_class[label_idx] = 1

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
        for nr, epoch in enumerate(range(epochs)):
            print(f"Epoch {nr + 1}/{epochs}")
            epoch_loss = 0
            for i, data in enumerate(dataloader):
                tens, label = data
                #epoch_loss = 0
                #print(tens, label)
                #print(label.item())
                #print(tens)

                # Set param-space gradient to zero
                optim.zero_grad()

                # Get model output
                #print("into self")
                out = self(tens)
                #print("self call successful")
                # Calculate loss
                #print("into loss")
                l = self.loss(out, label)
                print(out)
                print(label)
                print(l)
                print("")
                #print("loss: ", l.item())
                epoch_loss += l.item()
                #print("loss call successful")
                # Perform backpropagation
                l.backward()
                #print("backprop successful")
                # Step in param space
                optim.step()
                #print("optim successful")
            self.loss_epoch.append(epoch_loss/len(dataset))
            print("last avg. loss: ", self.loss_epoch[-1])
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
