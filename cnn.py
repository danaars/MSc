import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from cnn_setup import Pipesound
from torch.utils.data import DataLoader, random_split

abspath = "/run/media/daniel/ATLEJ-STT/data/test"

ds = Pipesound("meta.csv", abspath, transform = "pad", width_cut = 700)

l = len(ds)
trainsize = int(0.75*l)
testsize = int(l - trainsize)

train, test = random_split(ds, [trainsize, testsize])

trainloader = DataLoader(train, shuffle=True, num_workers=1)
testloader = DataLoader(test, shuffle=True, num_workers=1)

class FlowClassify(nn.Module):
    def __init__(self, sample_input, device):
        super(FlowClassify, self).__init__()
        self.sample_input = sample_input
        self.device = device

        # Define Conv layers and pooling
        self.conv1 = nn.Conv2d(1, 8, 16, device=self.device)    # 1 ch in, 8 filter out 16x16
        self.conv2 = nn.Conv2d(8, 16, 8, device=self.device)    # 8 ch in, 16, filters out, 8x8
        self.conv3 = nn.Conv2d(16, 32, 5, device=self.device)
        self.pool = nn.MaxPool2d(2, 2)        # 2x2, stride 2

        self.flattenlength = self._getflatlen()

        # Define Dense layers
        self.classify1 = nn.Linear(self.flattenlength, 128, device=self.device)
        self.classify2 = nn.Linear(128, 128, device=self.device)
        self.classify3 = nn.Linear(128, 2, device=self.device)

    def _getflatlen(self):
        with torch.no_grad():
            x = self._conv(self.sample_input)
            #print(x.shape)
            l = 1
            for i in range(len(x.shape)):
                l *= x.shape[i]
            #print(l)
        return l

    def _conv(self, tens):
        act = F.relu
        x = self.pool(act(self.conv1(tens)))
        x = self.pool(act(self.conv2(x)))
        x = self.pool(act(self.conv3(x)))
        return x

    def _classify(self, tens):
        x = torch.sigmoid(self.classify1(tens))
        x = torch.sigmoid(self.classify2(x))
        x = self.classify3(x)
        return x

    def forward(self, tens):
        #conv
        x = self._conv(tens)
        x = x.view(-1, self.flattenlength)

        #dense
        class_pred = self._classify(x)
        return class_pred
