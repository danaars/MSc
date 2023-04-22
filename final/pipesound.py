import os
import torch
import pandas as pd
import numpy as np

def pad(tens, maxshape):
    # Note: 2D only
    channels, height, width = tens.shape
    maxheight, maxwidth = maxshape

    w_diff = maxwidth - width
    h_diff = maxheight - height

    lft_pad = int(np.ceil(w_diff/2))
    rgt_pad = int(np.floor(w_diff/2))
    top_pad = int(np.ceil(h_diff/2))
    btm_pad = int(np.floor(h_diff/2))

    p = (lft_pad, rgt_pad, top_pad, btm_pad)
    tens = torch.nn.functional.pad(tens,
                                   pad = p,
                                   mode='constant')
    return tens

# Data set definiton
class PipeSound(torch.utils.data.Dataset):
    def __init__(self,
                 csv_file,
                 data_dir,
                 transform = None):

        self.data_table = pd.read_csv(os.path.join(data_dir, csv_file))
        self.data_dir = data_dir

        self.transform = transform

        self.lens = self.data_table['endind'] - self.data_table['startind'] + 1
        self.maxlen = self.lens.max()

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self,
                    idx,):

        df = self.data_table.iloc[idx]
        filename = df['filename']
        label_idx = df['label']
        velocity = df['velocity']
        length = df['length']

        file_pth = os.path.join(self.data_dir, filename)
        rawdata = np.loadtxt(file_pth)

        if self.transform:
            tens = self.transform(rawdata)

        tens = pad(tens, self.maxshape)

        return tens, label_idx, velocity, length

    def getsample(self,
                  inchannels,
                  device):
        return torch.rand( (inchannels, *self.maxshape), device = device)

    def set_transform(self, custom_transform):
        self.transform = custom_transform
        self.maxshape = self.transform(np.ones(self.maxlen)).shape
